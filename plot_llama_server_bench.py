"""Plot llama-server benchmark reports as per-model comparison graphs."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("plot_llama_server_bench")
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

SLUG_RE = re.compile(r"[^a-z0-9._-]+")

RATE_COLUMN = {
    "prefill": "prompt_tokens_per_second",
    "generation": "generation_tokens_per_second",
}


class PlotError(RuntimeError):
    """Raised when a report file cannot be turned into a graph."""


@dataclass(frozen=True)
class Sample:
    model: str
    build_id: str
    test: str
    depth: int
    rate: float


@dataclass
class Options:
    """Command line options; argparse writes directly into these fields."""

    reports: list[str] = field(default_factory=list)
    output_dir: str = "graphs"
    log_level: str = "INFO"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


@dataclass(frozen=True)
class Group:
    model: str
    build_id: str
    test: str
    depth: int
    mean: float
    sd: float


def model_slug(model: str) -> str:
    slug = SLUG_RE.sub("_", model.lower()).strip("_")
    return slug or "model"


def parse_args(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(
        description="Plot llama-server benchmark reports as comparison graphs."
    )
    parser.add_argument("reports", nargs="+", help="per-model report CSV paths")
    parser.add_argument("--output-dir", help="directory for the PNG files")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=LOG_LEVELS,
        help="logging verbosity",
    )
    # The defaults live on the dataclass; argparse keeps them and only overwrites
    # what the command line provides.
    options = Options()
    parser.parse_args(argv, namespace=options)
    return options


def read_report(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in CSV_HEADER if column not in (reader.fieldnames or [])]
        if missing:
            raise PlotError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)

    samples: list[Sample] = []
    for row in rows:
        model = row["model"]
        build_id = row["build_id"]
        test = row["test"]
        if test not in RATE_COLUMN:
            raise PlotError(f"{path}: unknown test {test!r}")
        raw_rate = row[RATE_COLUMN[test]]
        if not raw_rate:
            continue
        try:
            rate = float(raw_rate)
            depth = int(row["depth_tokens"])
        except ValueError as exc:
            raise PlotError(f"{path}: malformed row {row}") from exc
        samples.append(Sample(model, build_id, test, depth, rate))
    return samples


def group_samples(samples: Sequence[Sample]) -> list[Group]:
    buckets: dict[tuple[str, str, str, int], list[float]] = {}
    for sample in samples:
        key = (sample.model, sample.build_id, sample.test, sample.depth)
        buckets.setdefault(key, []).append(sample.rate)

    groups: list[Group] = []
    for (model, build_id, test, depth), rates in buckets.items():
        mean = statistics.mean(rates)
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        groups.append(Group(model, build_id, test, depth, mean, sd))
    return groups


def render(
    groups: Sequence[Group],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    models = sorted({group.model for group in groups})
    slugs: dict[str, str] = {}
    for model in models:
        slug = model_slug(model)
        if slug in slugs:
            raise PlotError(
                f"model names {slugs[slug]!r} and {model!r} map to the same file name {slug!r}"
            )
        slugs[slug] = model

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model in models:
        model_groups = [group for group in groups if group.model == model]
        depths = sorted({group.depth for group in model_groups})
        build_ids = sorted({group.build_id for group in model_groups})

        figure = pyplot.figure(figsize=(13, 5.5))
        prefill_axis = figure.add_subplot(1, 2, 1)
        generation_axis = figure.add_subplot(1, 2, 2)
        generation_axis.sharex(prefill_axis)
        panels = (
            (prefill_axis, "prefill", "Prefill tokens/s"),
            (generation_axis, "generation", "Generation tokens/s"),
        )

        for axis, test, ylabel in panels:
            for build_id in build_ids:
                points = sorted(
                    (group.depth, group.mean, group.sd)
                    for group in model_groups
                    if group.test == test and group.build_id == build_id
                )
                if not points:
                    continue
                axis.errorbar(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    yerr=[point[2] for point in points],
                    marker="o",
                    capsize=3,
                    label=build_id,
                )
            axis.set_xlabel("Starting context depth (tokens)")
            axis.set_ylabel(ylabel)
            axis.set_ylim(bottom=0)
            axis.set_xticks(depths)
            axis.set_xticklabels([f"{depth:,}" for depth in depths], rotation=30)
            axis.grid(True, alpha=0.3)
            if build_ids:
                axis.legend()

        figure.suptitle(model)
        figure.tight_layout()
        path = output_dir / f"{model_slug(model)}_benchmark.png"
        figure.savefig(path, dpi=160)
        pyplot.close(figure)
        written.append(path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    setup_logging(options.log_level)
    samples: list[Sample] = []
    for name in options.reports:
        path = Path(name)
        if not path.is_file():
            LOGGER.warning("report not found: %s", path)
            return 1
        LOGGER.info("reading %s", path)
        try:
            report_samples = read_report(path)
        except PlotError as exc:
            LOGGER.warning("%s", exc)
            return 1
        LOGGER.info("read %d samples from %s", len(report_samples), path)
        samples.extend(report_samples)

    if not samples:
        LOGGER.warning("no samples found in the given reports")
        return 1

    groups = group_samples(samples)
    LOGGER.info("grouped %d samples into %d groups", len(samples), len(groups))
    written = render(groups, Path(options.output_dir))
    for path in written:
        LOGGER.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
