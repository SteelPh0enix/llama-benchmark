#!/usr/bin/env -S uv run
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
    model_path: str
    build_id: str
    server_build: str
    test: str
    depth: int
    rate: float


@dataclass
class Options:
    """Command line options; argparse writes directly into these fields."""

    reports: list[str] = field(default_factory=list)
    output_dir: str = "graphs"
    log_level: str = "INFO"
    compare_reports: list[str] = field(default_factory=list)


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
    server_build: str
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
    parser.add_argument("reports", nargs="*", help="per-model report CSV paths")
    parser.add_argument("--output-dir", help="directory for the PNG files")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=LOG_LEVELS,
        help="logging verbosity",
    )
    parser.add_argument(
        "--compare-builds",
        dest="compare_reports",
        nargs=2,
        metavar=("REPORT_A", "REPORT_B"),
        help="plot the two builds described by these report CSVs onto a single figure",
    )
    # The defaults live on the dataclass; argparse keeps them and only overwrites
    # what the command line provides.
    options = Options()
    parser.parse_args(argv, namespace=options)
    if not options.reports and not options.compare_reports:
        parser.error("no report paths given: pass report paths or --compare-builds A.csv B.csv")
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
        model_path = row["model_path"]
        build_id = row["build_id"]
        server_build = row["server_build"]
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
        samples.append(
            Sample(
                model=model,
                model_path=model_path,
                build_id=build_id,
                server_build=server_build,
                test=test,
                depth=depth,
                rate=rate,
            )
        )
    return samples


def group_samples(samples: Sequence[Sample]) -> list[Group]:
    buckets: dict[tuple[str, str, str, int], list[float]] = {}
    server_builds: dict[tuple[str, str, str, int], set[str]] = {}
    for sample in samples:
        key = (sample.model, sample.build_id, sample.test, sample.depth)
        buckets.setdefault(key, []).append(sample.rate)
        server_builds.setdefault(key, set()).add(sample.server_build)

    groups: list[Group] = []
    for (model, build_id, test, depth), rates in buckets.items():
        mean = statistics.mean(rates)
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        groups.append(
            Group(
                model=model,
                build_id=build_id,
                server_build="+".join(sorted(server_builds[(model, build_id, test, depth)])),
                test=test,
                depth=depth,
                mean=mean,
                sd=sd,
            )
        )
    return groups


def report_build_id(path: Path, samples: Sequence[Sample]) -> str:
    """Return the build ID a report was written by; raise PlotError when it is not unique."""
    build_ids = sorted({sample.build_id for sample in samples})
    if not build_ids:
        raise PlotError(f"{path}: no samples found")
    if len(build_ids) > 1:
        raise PlotError(f"{path}: report covers more than one build: {', '.join(build_ids)}")
    return build_ids[0]


def validate_comparison(samples: Sequence[Sample], build_a: str, build_b: str) -> str:
    """Return the shared model name; raise PlotError when the builds are not comparable."""
    if build_a == build_b:
        raise PlotError(
            f"the two compared reports must come from different builds: both report {build_a!r}"
        )

    available = sorted({sample.build_id for sample in samples})
    for build in (build_a, build_b):
        if not any(sample.build_id == build for sample in samples):
            raise PlotError(
                f"no samples for build {build!r}; available builds: {', '.join(available)}"
            )

    models: dict[str, tuple[str, str]] = {}
    for build in (build_a, build_b):
        pairs = {
            (sample.model, sample.model_path) for sample in samples if sample.build_id == build
        }
        if len(pairs) > 1:
            names = sorted(f"{model} ({path})" for model, path in pairs)
            raise PlotError(f"build {build!r} covers more than one model: {', '.join(names)}")
        models[build] = next(iter(pairs))

    model_a, path_a = models[build_a]
    model_b, path_b = models[build_b]
    if model_a != model_b:
        raise PlotError(
            f"builds {build_a!r} and {build_b!r} report different models: "
            + f"{model_a!r} and {model_b!r}"
        )
    if path_a != path_b:
        raise PlotError(
            f"builds {build_a!r} and {build_b!r} report different model paths: "
            + f"{path_a!r} and {path_b!r}"
        )
    return model_a


def render(
    groups: Sequence[Group],
    output_dir: Path,
    *,
    series_order: Sequence[str] | None = None,
    compare_name: str | None = None,
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
        if series_order is not None:
            build_ids = [build_id for build_id in series_order if build_id in build_ids]
        labels: dict[str, str] = {}
        for build_id in build_ids:
            builds = sorted(
                {group.server_build for group in model_groups if group.build_id == build_id}
            )
            label = build_id if compare_name is None else f"{build_id} ({'+'.join(builds)})"
            labels[build_id] = label

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
                    label=labels[build_id],
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
        path = output_dir / (
            f"{model_slug(model)}__{compare_name}.png"
            if compare_name is not None
            else f"{model_slug(model)}_benchmark.png"
        )
        figure.savefig(path, dpi=160)
        pyplot.close(figure)
        written.append(path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    setup_logging(options.log_level)

    compare_paths = [Path(name) for name in options.compare_reports]
    if len(compare_paths) == 2 and compare_paths[0] == compare_paths[1]:
        LOGGER.warning(
            "the two --compare-builds reports must be different files: %s", compare_paths[0]
        )
        return 1

    # The compared reports come first; every path is read once.
    entries: dict[Path, bool] = {path: True for path in compare_paths}
    for name in options.reports:
        entries.setdefault(Path(name), False)

    samples: list[Sample] = []
    compared: list[str] = []
    for path, is_compared in entries.items():
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
        if is_compared:
            try:
                compared.append(report_build_id(path, report_samples))
            except PlotError as exc:
                LOGGER.warning("%s", exc)
                return 1

    if not samples:
        LOGGER.warning("no samples found in the given reports")
        return 1

    series_order: list[str] | None = None
    compare_name: str | None = None
    if compared:
        build_a, build_b = compared
        try:
            model = validate_comparison(samples, build_a, build_b)
        except PlotError as exc:
            LOGGER.warning("%s", exc)
            return 1
        LOGGER.info("comparing builds %s and %s of model %s", build_a, build_b, model)
        samples = [sample for sample in samples if sample.build_id in (build_a, build_b)]
        series_order = [build_a, build_b]
        compare_name = f"{build_a}_vs_{build_b}"

    groups = group_samples(samples)
    LOGGER.info("grouped %d samples into %d groups", len(samples), len(groups))
    written = render(
        groups,
        Path(options.output_dir),
        series_order=series_order,
        compare_name=compare_name,
    )
    for path in written:
        LOGGER.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
