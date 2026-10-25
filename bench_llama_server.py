#!/usr/bin/env -S uv run
"""Benchmark llama-server prefill and generation throughput across context depths.

One invocation targets one running llama-server and one model preset. It measures
prefill throughput and generation throughput at a set of starting context depths,
writing one CSV and one Markdown report per model/server.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import TextIO, cast

JSONDict = dict[str, object]

LOGGER = logging.getLogger("bench_llama_server")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

CSV_HEADER = [
    "build_id",
    "server_build",
    "model",
    "model_path",
    "model_n_ctx",
    "test",
    "depth_tokens",
    "repetition",
    "prompt_n",
    "cache_n",
    "predicted_n",
    "prompt_ms",
    "predicted_ms",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "wall_ms",
]

BUILD_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SLUG_RE = re.compile(r"[^a-z0-9._-]+")
DEFAULT_DEPTHS = "0,8192,16384,32768,65536,131072,250000"

# The synthetic prompts are drawn from the model's own tokenizer. The corpus is
# generated from the seed alone, so every build of a given model sees byte-identical
# prompts. The pool must be large enough that the prompts carry real entropy
# instead of a handful of repeated pieces.
TOKEN_CORPUS_WORDS = 512
MIN_TOKEN_POOL = 512
TOKEN_CORPUS_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
TOKEN_CORPUS_SEPARATORS = " -_./"


# The server decides how much of a cached prefix it can reuse: plain attention
# models re-evaluate the last token, hybrid-attention models re-evaluate a small
# tail. The depth is still exact (cache_n + prompt_n == the prompt we sent), so the
# tail size is the only model-specific part and is bounded rather than guessed.
CACHE_TAIL_TOLERANCE = 64


class BenchmarkError(RuntimeError):
    """Raised for every recoverable failure of a benchmark run."""


def random_seed() -> int:
    """Pick a fresh seed so each run measures different prompts.

    The seed is recorded in the report; pass the same --seed to reproduce a run.
    """
    return random.SystemRandom().randrange(1, 2**31 - 1)


@dataclass
class Options:
    """Command line options; argparse writes directly into these fields."""

    url: str = "http://127.0.0.1:8080"
    build_id: str = ""
    model: str = ""
    output_dir: str = "results"
    depths: str = DEFAULT_DEPTHS
    prefill_tokens: int = 10240
    generation_tokens: int = 1024
    repetitions: int = 3
    seed: int = field(default_factory=random_seed)
    timeout: float = 3600.0
    api_key: str | None = None
    build_info: str | None = None
    log_level: str = "INFO"


@dataclass(frozen=True)
class Config:
    url: str
    build_id: str
    model: str
    output_dir: Path
    depths: tuple[int, ...]
    prefill_tokens: int
    generation_tokens: int
    repetitions: int
    seed: int
    timeout: float
    api_key: str | None
    build_info: str | None
    log_level: str


@dataclass(frozen=True)
class Timing:
    prompt_n: int
    cache_n: int
    predicted_n: int
    prompt_ms: float
    predicted_ms: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    truncated: bool

    def describe(self) -> str:
        return (
            f"prompt_n={self.prompt_n} cache_n={self.cache_n} "
            f"predicted_n={self.predicted_n} truncated={self.truncated}"
        )


@dataclass(frozen=True)
class Row:
    build_id: str
    server_build: str
    model: str
    model_path: str
    model_n_ctx: int
    test: str
    depth_tokens: int
    repetition: int
    prompt_n: int
    cache_n: int
    predicted_n: int
    prompt_ms: float
    predicted_ms: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float | None
    wall_ms: float

    def as_csv(self) -> list[str]:
        rate = (
            ""
            if self.generation_tokens_per_second is None
            else _fmt_float(self.generation_tokens_per_second)
        )
        return [
            self.build_id,
            self.server_build,
            self.model,
            self.model_path,
            str(self.model_n_ctx),
            self.test,
            str(self.depth_tokens),
            str(self.repetition),
            str(self.prompt_n),
            str(self.cache_n),
            str(self.predicted_n),
            _fmt_float(self.prompt_ms),
            _fmt_float(self.predicted_ms),
            _fmt_float(self.prompt_tokens_per_second),
            rate,
            _fmt_float(self.wall_ms),
        ]


def _fmt_float(value: float) -> str:
    """Shortest representation that round-trips back to the same float."""
    return repr(float(value))


def model_slug(model: str) -> str:
    slug = SLUG_RE.sub("_", model.lower()).strip("_")
    return slug or "model"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def parse_args(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server prefill and generation throughput."
    )
    parser.add_argument("--url", help="llama-server base URL")
    parser.add_argument("--build-id", required=True, help="series label of this build")
    parser.add_argument("--model", required=True, help="model preset name")
    parser.add_argument("--output-dir", help="directory for the report pair")
    parser.add_argument("--depths", help="comma-separated starting depths")
    parser.add_argument("--prefill-tokens", type=int)
    parser.add_argument("--generation-tokens", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--api-key", default=None, help="Bearer token for the server")
    parser.add_argument(
        "--build-info",
        default=None,
        help=(
            "override the reported llama.cpp build info (the server reports a placeholder "
            "like 'b0-unknown' when the binary was built without git metadata)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=LOG_LEVELS,
        help="logging verbosity (DEBUG also prints the raw server timings)",
    )
    # The defaults live on the dataclass; argparse keeps them and only overwrites
    # what the command line provides.
    options = Options()
    parser.parse_args(argv, namespace=options)
    return options


def build_config(options: Options) -> Config:
    build_id = options.build_id
    if not BUILD_ID_RE.fullmatch(build_id):
        raise BenchmarkError(f"invalid --build-id {build_id!r}: expected [A-Za-z0-9._-]+")
    model = options.model
    if not model.strip():
        raise BenchmarkError("--model must not be empty")

    depths_text = options.depths
    depths: list[int] = []
    for part in depths_text.split(","):
        part = part.strip()
        if not part:
            raise BenchmarkError(f"invalid --depths value {depths_text!r}")
        try:
            depths.append(int(part))
        except ValueError as exc:
            raise BenchmarkError(f"invalid --depths value {part!r}") from exc
    if not depths:
        raise BenchmarkError("--depths must list at least one depth")
    if any(depth < 0 for depth in depths):
        raise BenchmarkError("--depths must not contain negative depths")
    if depths != sorted(set(depths)):
        raise BenchmarkError("--depths must be strictly increasing")

    prefill_tokens = options.prefill_tokens
    generation_tokens = options.generation_tokens
    repetitions = options.repetitions
    timeout = options.timeout
    if prefill_tokens <= 0:
        raise BenchmarkError("--prefill-tokens must be positive")
    if generation_tokens <= 0:
        raise BenchmarkError("--generation-tokens must be positive")
    if repetitions < 1:
        raise BenchmarkError("--repetitions must be at least 1")
    if timeout <= 0:
        raise BenchmarkError("--timeout must be positive")

    return Config(
        url=options.url.rstrip("/"),
        build_id=build_id,
        model=model,
        output_dir=Path(options.output_dir),
        depths=tuple(depths),
        prefill_tokens=prefill_tokens,
        generation_tokens=generation_tokens,
        repetitions=repetitions,
        seed=options.seed,
        timeout=timeout,
        api_key=options.api_key,
        build_info=options.build_info,
        log_level=options.log_level,
    )


def http_json(
    url: str,
    payload: JSONDict | None,
    api_key: str | None,
    timeout: float,
) -> JSONDict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        response = cast("HTTPResponse", urllib.request.urlopen(request, timeout=timeout))
        with response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"{url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"{url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise BenchmarkError(f"{url} timed out after {timeout} s") from exc

    parsed = cast("object", json.loads(body))
    if not isinstance(parsed, dict):
        raise BenchmarkError(f"{url} returned a non-object JSON response")
    return {str(key): value for key, value in cast("dict[str, object]", parsed).items()}


def get_json(url: str, api_key: str | None, timeout: float) -> JSONDict:
    return http_json(url, None, api_key, timeout)


def require_str(mapping: JSONDict, key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{context}: expected a non-empty string for {key!r}")
    return value


def require_int(mapping: JSONDict, key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{context}: expected an integer for {key!r}")
    return value


def opt_str(mapping: JSONDict, key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def opt_float(mapping: JSONDict, key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def opt_mapping(mapping: JSONDict, key: str) -> JSONDict:
    value = mapping.get(key)
    if not isinstance(value, dict):
        return {}
    return dict(cast("dict[str, object]", value))


def str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in cast("list[object]", value) if isinstance(item, str))


def fetch_props(config: Config) -> tuple[bool, JSONDict]:
    base_props = get_json(f"{config.url}/props", config.api_key, config.timeout)
    if base_props.get("role") != "router":
        return False, base_props
    query = urllib.parse.urlencode({"model": config.model})
    model_props = get_json(f"{config.url}/props?{query}", config.api_key, config.timeout)
    return True, model_props


def build_token_corpus(seed: int, words: int) -> str:
    """Build a deterministic corpus of random words for tokenization.

    The text depends only on the seed, so every build of a given model tokenizes
    the same corpus into the same prompt tokens.
    """
    rng = random.Random(f"{seed}:token-corpus:{words}")
    out: list[str] = []
    for _ in range(words):
        segments = [
            "".join(rng.choice(TOKEN_CORPUS_ALPHABET) for _ in range(rng.randint(2, 10)))
            for _ in range(rng.randint(1, 5))
        ]
        separator = rng.choice(TOKEN_CORPUS_SEPARATORS)
        out.append(separator.join(segments))
    return " ".join(out)


def tokenize_corpus(config: Config, corpus: str) -> list[int]:
    payload = http_json(
        f"{config.url}/tokenize",
        {
            "model": config.model,
            "content": corpus,
            "add_special": False,
            "parse_special": False,
        },
        config.api_key,
        config.timeout,
    )
    raw = payload.get("tokens")
    if not isinstance(raw, list):
        raise BenchmarkError("/tokenize did not return a token list")

    pool: list[int] = []
    for item in cast("list[object]", raw):
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if item not in pool:
            pool.append(item)
    return pool


def fetch_token_pool(config: Config) -> list[int]:
    """Collect at least MIN_TOKEN_POOL distinct token ids from the model's tokenizer.

    The corpus grows until the tokenizer yields enough distinct pieces: a small
    vocabulary splits short pseudo-words into a handful of repeated pieces, which
    would make the prompts far less varied than intended.
    """
    words = TOKEN_CORPUS_WORDS
    best: list[int] = []
    for attempt in range(4):
        pool = tokenize_corpus(config, build_token_corpus(config.seed, words))
        LOGGER.debug("tokenized a %d-word corpus into %d distinct ids", words, len(pool))
        if len(pool) > len(best):
            best = pool
        if len(pool) >= MIN_TOKEN_POOL:
            if attempt:
                LOGGER.warning(
                    "token pool needed %d retries to reach %d distinct ids", attempt + 1, len(pool)
                )
            return pool
        LOGGER.warning(
            "token pool has only %d distinct ids (need %d); retrying with a %d-word corpus",
            len(pool),
            MIN_TOKEN_POOL,
            words * 4,
        )
        words *= 4
    raise BenchmarkError(
        f"model {config.model} yields only {len(best)} distinct token ids from a "
        + f"{words // 4}-word corpus, need at least {MIN_TOKEN_POOL}"
    )


def timed_completion(
    config: Config,
    prompt: list[int],
    n_predict: int,
    cache_prompt: bool,
    ignore_eos: bool,
) -> tuple[Timing, float]:
    payload: JSONDict = {
        "model": config.model,
        "id_slot": 0,
        "stream": False,
        "return_tokens": False,
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": cache_prompt,
        "ignore_eos": ignore_eos,
        "temperature": 0,
        "top_k": 1,
        "seed": config.seed,
    }
    started = time.perf_counter()
    response = http_json(f"{config.url}/completion", payload, config.api_key, config.timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return extract_timing(response), wall_ms


def extract_timing(response: JSONDict) -> Timing:
    timings = opt_mapping(response, "timings")
    if not timings:
        raise BenchmarkError("/completion response without timings")
    return Timing(
        prompt_n=require_int(timings, "prompt_n", "timings"),
        cache_n=require_int(timings, "cache_n", "timings"),
        predicted_n=require_int(timings, "predicted_n", "timings"),
        prompt_ms=opt_float(timings, "prompt_ms") or 0.0,
        predicted_ms=opt_float(timings, "predicted_ms") or 0.0,
        prompt_tokens_per_second=opt_float(timings, "prompt_per_second") or 0.0,
        generation_tokens_per_second=opt_float(timings, "predicted_per_second") or 0.0,
        truncated=bool(response.get("truncated")),
    )


def expect(
    label: str,
    timing: Timing,
    config: Config,
    depth: int,
    repetition: int,
    expected_total: int,
    min_cache_n: int,
    max_prompt_n: int,
) -> None:
    """Validate one measurement against the cache invariants the server guarantees.

    The prompt we sent is fully accounted for by cache_n + prompt_n, the cached
    prefix was actually reused, and only a bounded tail was re-evaluated.
    """
    total = timing.cache_n + timing.prompt_n
    problems: list[str] = []
    if total != expected_total:
        problems.append(f"cache_n + prompt_n == {total}, expected {expected_total}")
    if timing.cache_n < min_cache_n:
        problems.append(f"cache_n == {timing.cache_n}, expected at least {min_cache_n}")
    if timing.prompt_n > max_prompt_n:
        problems.append(f"prompt_n == {timing.prompt_n}, expected at most {max_prompt_n}")
    if timing.truncated:
        problems.append("the server truncated the prompt")
    if not problems:
        return
    raise BenchmarkError(
        f"{label} mismatch (model={config.model} depth={depth} repetition={repetition}): "
        + "; ".join(problems)
        + f" [{timing.describe()}]"
    )


def run_sweep(
    config: Config,
    server_build: str,
    model_path: str,
    model_n_ctx: int,
    token_pool: list[int],
    on_row: Callable[[Row], None],
) -> None:
    def build_prefix(depth: int, repetition: int) -> list[int]:
        rng = random.Random(f"{config.seed}:{config.model}:{repetition}:prefix:{depth}")
        return [rng.choice(token_pool) for _ in range(depth)]

    def build_suffix(depth: int, repetition: int) -> list[int]:
        rng = random.Random(f"{config.seed}:{config.model}:{repetition}:prefill-suffix:{depth}")
        return [rng.choice(token_pool) for _ in range(config.prefill_tokens)]

    def make_row(
        test: str,
        depth: int,
        repetition: int,
        timing: Timing,
        wall_ms: float,
    ) -> Row:
        return Row(
            build_id=config.build_id,
            server_build=server_build,
            model=config.model,
            model_path=model_path,
            model_n_ctx=model_n_ctx,
            test=test,
            depth_tokens=depth,
            repetition=repetition,
            prompt_n=timing.prompt_n,
            cache_n=timing.cache_n,
            predicted_n=timing.predicted_n,
            prompt_ms=timing.prompt_ms,
            predicted_ms=timing.predicted_ms,
            prompt_tokens_per_second=timing.prompt_tokens_per_second,
            generation_tokens_per_second=(
                None if test == "prefill" else timing.generation_tokens_per_second
            ),
            wall_ms=wall_ms,
        )

    for repetition in range(1, config.repetitions + 1):
        LOGGER.info("repetition %d of %d", repetition, config.repetitions)
        for depth in config.depths:
            LOGGER.info("depth %s: seeding a %d token context", f"{depth:,}", depth)
            prefix = build_prefix(depth, repetition)
            suffix = build_suffix(depth, repetition)

            if depth > 0:
                seed_timing, seed_wall = timed_completion(
                    config, prefix, 1, cache_prompt=False, ignore_eos=False
                )
                LOGGER.debug(
                    "depth %d rep %d seed: %s wall=%.1f ms",
                    depth,
                    repetition,
                    seed_timing.describe(),
                    seed_wall,
                )
                expect(
                    "prefill seed",
                    seed_timing,
                    config,
                    depth,
                    repetition,
                    expected_total=depth,
                    min_cache_n=0,
                    max_prompt_n=depth,
                )
                if seed_timing.predicted_n != 1:
                    raise BenchmarkError(
                        f"prefill seed predicted {seed_timing.predicted_n} tokens, expected 1"
                    )

            prefill_timing, prefill_wall = timed_completion(
                config,
                prefix + suffix,
                1,
                cache_prompt=depth > 0,
                ignore_eos=False,
            )
            LOGGER.debug(
                "depth %d rep %d prefill: %s prompt_ms=%.2f rate=%.2f tok/s wall=%.1f ms",
                depth,
                repetition,
                prefill_timing.describe(),
                prefill_timing.prompt_ms,
                prefill_timing.prompt_tokens_per_second,
                prefill_wall,
            )
            expect(
                "prefill",
                prefill_timing,
                config,
                depth,
                repetition,
                expected_total=depth + config.prefill_tokens,
                min_cache_n=max(depth - CACHE_TAIL_TOLERANCE, 0),
                max_prompt_n=config.prefill_tokens + CACHE_TAIL_TOLERANCE,
            )
            if prefill_timing.predicted_n != 1:
                raise BenchmarkError(
                    f"prefill predicted {prefill_timing.predicted_n} tokens, expected 1"
                )
            prefill_row = make_row("prefill", depth, repetition, prefill_timing, prefill_wall)
            on_row(prefill_row)
            LOGGER.info(
                "prefill  depth=%7d rep=%d cache_n=%d prompt_n=%d rate=%s tok/s wall=%s ms",
                depth,
                repetition,
                prefill_row.cache_n,
                prefill_row.prompt_n,
                f"{prefill_row.prompt_tokens_per_second:,.2f}",
                f"{prefill_row.wall_ms:,.1f}",
            )

            generation_prompt = prefix if depth > 0 else [token_pool[0]]
            generation_timing, generation_wall = timed_completion(
                config,
                generation_prompt,
                config.generation_tokens,
                cache_prompt=depth > 0,
                ignore_eos=True,
            )
            LOGGER.debug(
                "depth %d rep %d generation: %s predicted_ms=%.2f rate=%.2f tok/s wall=%.1f ms",
                depth,
                repetition,
                generation_timing.describe(),
                generation_timing.predicted_ms,
                generation_timing.generation_tokens_per_second,
                generation_wall,
            )
            expect(
                "generation",
                generation_timing,
                config,
                depth,
                repetition,
                expected_total=depth if depth > 0 else 1,
                min_cache_n=max(depth - CACHE_TAIL_TOLERANCE, 0),
                max_prompt_n=CACHE_TAIL_TOLERANCE if depth > 0 else 1,
            )
            if generation_timing.predicted_n != config.generation_tokens:
                raise BenchmarkError(
                    f"generation predicted {generation_timing.predicted_n} tokens, "
                    + f"expected {config.generation_tokens}"
                )
            generation_row = make_row(
                "generation", depth, repetition, generation_timing, generation_wall
            )
            on_row(generation_row)
            LOGGER.info(
                "generation depth=%7d rep=%d cache_n=%d predicted_n=%d rate=%s tok/s wall=%s ms",
                depth,
                repetition,
                generation_row.cache_n,
                generation_row.predicted_n,
                f"{generation_row.generation_tokens_per_second or 0.0:,.2f}",
                f"{generation_row.wall_ms:,.1f}",
            )


def write_csv_header(handle: TextIO) -> None:
    csv.writer(handle).writerow(CSV_HEADER)
    handle.flush()


def write_csv_row(handle: TextIO, row: Row) -> None:
    csv.writer(handle).writerow(row.as_csv())
    handle.flush()


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values)


def _sd_cell(values: Sequence[float]) -> str:
    if len(values) < 2:
        return "n/a"
    return f"{statistics.stdev(values):.2f}"


def write_markdown(
    path: Path,
    config: Config,
    props: JSONDict,
    server_build: str,
    model_path: str,
    model_n_ctx: int,
    rows: Sequence[Row],
) -> None:
    model_alias = opt_str(props, "model_alias")
    ftype = opt_str(props, "model_ftype")

    lines: list[str] = [
        f"# llama-server benchmark: {config.build_id} / {config.model}",
        "",
        f"- Date (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- URL: {config.url}",
        f"- Server build: {server_build}",
        f"- Seed: {config.seed}",
        f"- Depths: {', '.join(f'{depth:,}' for depth in config.depths)}",
        (
            f"- Prefill tokens: {config.prefill_tokens} / generation tokens: "
            f"{config.generation_tokens} / repetitions: {config.repetitions}"
        ),
        "- Runtime note: generation depth 0 uses a one-token seed prompt",
        "",
        "## Model details",
        "",
        "Details come from `/props` captured before the sweep.",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]

    detail_rows = [
        ("Model", model_alias or "n/a"),
        ("Quantization", ftype or "n/a"),
        ("Model path", model_path),
        ("Per-slot context", f"{model_n_ctx:,}"),
        ("llama.cpp build", server_build),
    ]
    lines.extend(f"| {name} | {value} |" for name, value in detail_rows)
    lines.append("")

    lines.append("## Prefill")
    lines.append("")
    lines.append(
        "| Depth | Prompt evaluated | Cache reused | Mean prefill tok/s | SD | Mean prefill ms |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for depth in config.depths:
        samples = [row for row in rows if row.test == "prefill" and row.depth_tokens == depth]
        if not samples:
            continue
        rates = [row.prompt_tokens_per_second for row in samples]
        durations = [row.prompt_ms for row in samples]
        prompt_n = _mean([float(row.prompt_n) for row in samples])
        cache_n = _mean([float(row.cache_n) for row in samples])
        lines.append(
            f"| {depth:,} | {prompt_n:.0f} | {cache_n:.0f} | {_mean(rates):.2f} | "
            + f"{_sd_cell(rates)} | {_mean(durations):.2f} |"
        )
    lines.append("")

    lines.append("## Generation")
    lines.append("")
    lines.append("| Depth | Predicted tokens | Mean generation tok/s | SD | Mean generation ms |")
    lines.append("| --- | --- | --- | --- | --- |")
    for depth in config.depths:
        samples = [row for row in rows if row.test == "generation" and row.depth_tokens == depth]
        if not samples:
            continue
        rates = [
            row.generation_tokens_per_second
            for row in samples
            if row.generation_tokens_per_second is not None
        ]
        durations = [row.predicted_ms for row in samples]
        predicted_n = _mean([float(row.predicted_n) for row in samples])
        lines.append(
            f"| {depth:,} | {predicted_n:.0f} | {_mean(rates):.2f} | "
            + f"{_sd_cell(rates)} | {_mean(durations):.2f} |"
        )
    lines.append("")

    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def unload_model(config: Config, tolerant: bool) -> None:
    LOGGER.info("unloading model '%s'", config.model)
    try:
        http_json(
            f"{config.url}/models/unload",
            {"model": config.model},
            config.api_key,
            config.timeout,
        )
    except BenchmarkError as exc:
        if not tolerant:
            raise
        LOGGER.warning("could not unload model '%s': %s", config.model, exc)
        return

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        payload = get_json(f"{config.url}/models", config.api_key, config.timeout)
        entries = payload.get("data")
        entry: JSONDict | None = None
        if isinstance(entries, list):
            for item in cast("list[object]", entries):
                if not isinstance(item, dict):
                    continue
                candidate = dict(cast("dict[str, object]", item))
                if candidate.get("id") == config.model or config.model in str_tuple(
                    candidate.get("aliases")
                ):
                    entry = candidate
                    break
        if entry is None:
            LOGGER.info("model '%s' is no longer listed", config.model)
            return
        if opt_mapping(entry, "status").get("value") == "unloaded":
            LOGGER.info("model '%s' is unloaded", config.model)
            return
        time.sleep(1.0)

    if not tolerant:
        raise BenchmarkError(f"model {config.model} was still loaded after 60 s")
    LOGGER.warning("model '%s' was still loaded after 60 s", config.model)


def run(config: Config) -> int:
    slug = model_slug(config.model)
    csv_path = config.output_dir / f"{slug}__{config.build_id}.csv"
    md_path = config.output_dir / f"{slug}__{config.build_id}.md"
    for path in (csv_path, md_path):
        if path.exists():
            raise BenchmarkError(f"refusing to overwrite existing file: {path}")

    LOGGER.info(
        "benchmark '%s' against %s (model '%s', seed %d)",
        config.build_id,
        config.url,
        config.model,
        config.seed,
    )
    LOGGER.info(
        "workload: depths=[%s] prefill_tokens=%d generation_tokens=%d repetitions=%d seed=%d",
        ", ".join(f"{depth:,}" for depth in config.depths),
        config.prefill_tokens,
        config.generation_tokens,
        config.repetitions,
        config.seed,
    )
    LOGGER.info("writing %s and %s", csv_path, md_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    router_mode, props = fetch_props(config)
    server_build = config.build_info or require_str(props, "build_info", "/props")
    model_path = require_str(props, "model_path", "/props")
    default_settings = opt_mapping(props, "default_generation_settings")
    model_n_ctx = require_int(default_settings, "n_ctx", "/props default_generation_settings")

    LOGGER.info("server mode: %s", "router" if router_mode else "direct")
    LOGGER.info("server build: %s", server_build)
    LOGGER.info("model path: %s", model_path)
    if config.build_info is None and server_build.endswith("-unknown"):
        LOGGER.warning(
            "the server reports a placeholder build info ('%s'); "
            + "pass --build-info to record the real commit",
            server_build,
        )
    elif config.build_info is not None:
        LOGGER.debug("build info overridden by --build-info")

    required_context = (
        max(
            max(config.depths) + max(config.prefill_tokens, config.generation_tokens),
            config.prefill_tokens + 32,
        )
        + 2
    )
    LOGGER.info(
        "context: %d tokens per slot available, %d required for this workload",
        model_n_ctx,
        required_context,
    )
    if required_context > model_n_ctx:
        raise BenchmarkError(
            f"model {config.model} needs a context of at least {required_context} tokens "
            + f"but only {model_n_ctx} are available"
        )

    token_pool = fetch_token_pool(config)
    LOGGER.info("token pool: %d distinct ids", len(token_pool))

    LOGGER.info("warming up with %d tokens", config.prefill_tokens)
    warmup_rng = random.Random(f"{config.seed}:warmup")
    warmup_prompt = [warmup_rng.choice(token_pool) for _ in range(config.prefill_tokens)]
    # The warm-up result is deliberately discarded.
    _, warmup_wall = timed_completion(
        config, warmup_prompt, 32, cache_prompt=False, ignore_eos=True
    )
    LOGGER.info("warmup finished in %.1f ms", warmup_wall)

    rows: list[Row] = []
    started = time.monotonic()

    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        write_csv_header(handle)

        def on_row(row: Row) -> None:
            rows.append(row)
            write_csv_row(handle, row)

        try:
            run_sweep(
                config,
                server_build,
                model_path,
                model_n_ctx,
                token_pool,
                on_row,
            )
        finally:
            if router_mode:
                unload_model(config, tolerant=True)

    write_markdown(
        md_path,
        config,
        props,
        server_build,
        model_path,
        model_n_ctx,
        rows,
    )
    LOGGER.info("wrote %s", csv_path)
    LOGGER.info("wrote %s", md_path)
    LOGGER.info("sweep finished in %.1f s", time.monotonic() - started)

    LOGGER.info("summary (mean over %d repetitions)", config.repetitions)
    for depth in config.depths:
        prefill_rates = [
            row.prompt_tokens_per_second
            for row in rows
            if row.test == "prefill" and row.depth_tokens == depth
        ]
        generation_rates = [
            row.generation_tokens_per_second or 0.0
            for row in rows
            if row.test == "generation" and row.depth_tokens == depth
        ]
        if not prefill_rates or not generation_rates:
            continue
        LOGGER.info(
            "  depth %7d | prefill %9s tok/s | generation %8s tok/s",
            depth,
            f"{statistics.mean(prefill_rates):,.2f}",
            f"{statistics.mean(generation_rates):,.2f}",
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    setup_logging(options.log_level)
    try:
        config = build_config(options)
        return run(config)
    except KeyboardInterrupt:
        LOGGER.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
