import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _jsonable(
    value: Any,
) -> Any:
    """Convert Pydantic/container values into JSON-safe data."""

    if isinstance(
        value,
        BaseModel,
    ):
        return value.model_dump(
            mode="json"
        )

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _jsonable(item)
            for item in value
        ]

    return value


def _csv_value(
    value: Any,
) -> str | int | float | bool | None:
    """Convert a value into a stable CSV-compatible representation."""

    value = _jsonable(
        value
    )

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _report_payload(
    report: BaseModel,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the canonical JSON payload for one evaluation report."""

    payload = {
        "report_type": (
            report.__class__.__name__
        ),
        "metadata": _jsonable(
            metadata or {}
        ),
        "report": report.model_dump(
            mode="json"
        ),
    }

    return payload


def save_report_json(
    report: BaseModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a complete evaluation report as formatted JSON."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = _report_payload(
        report=report,
        metadata=metadata,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def save_report_summary_csv(
    report: BaseModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Write top-level scalar report metrics as a two-column CSV.

    Nested per-case results are intentionally excluded here and are
    written separately by ``save_report_cases_csv``.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = report.model_dump(
        mode="json"
    )

    rows: list[
        tuple[str, str | int | float | bool | None]
    ] = [
        (
            "report_type",
            report.__class__.__name__,
        )
    ]

    for key, value in (
        metadata or {}
    ).items():
        rows.append(
            (
                f"metadata.{key}",
                _csv_value(value),
            )
        )

    for key, value in report_data.items():
        if key == "cases":
            continue

        rows.append(
            (
                key,
                _csv_value(value),
            )
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        writer.writerows(
            rows
        )

    return path


def save_report_cases_csv(
    report: BaseModel,
    path: str | Path,
) -> Path:
    """Write per-case evaluation rows as CSV."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = report.model_dump(
        mode="json"
    )

    raw_cases = report_data.get(
        "cases",
        [],
    )

    if not isinstance(
        raw_cases,
        list,
    ):
        raise ValueError(
            "Evaluation report 'cases' field "
            "must be a list."
        )

    cases: list[
        dict[str, Any]
    ] = []

    for raw_case in raw_cases:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise ValueError(
                "Each evaluation case result "
                "must be an object."
            )

        cases.append(
            raw_case
        )

    fieldnames: list[str] = []

    for case in cases:
        for key in case:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        if not fieldnames:
            file.write("")
            return path

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for case in cases:
            writer.writerow(
                {
                    key: _csv_value(
                        case.get(
                            key
                        )
                    )
                    for key in fieldnames
                }
            )

    return path


def save_report_summary_text(
    report: BaseModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a compact human-readable report summary."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = report.model_dump(
        mode="json"
    )

    lines = [
        (
            "report_type: "
            f"{report.__class__.__name__}"
        )
    ]

    for key, value in (
        metadata or {}
    ).items():
        lines.append(
            (
                f"metadata.{key}: "
                f"{_csv_value(value)}"
            )
        )

    for key, value in report_data.items():
        if key == "cases":
            continue

        lines.append(
            f"{key}: {_csv_value(value)}"
        )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def save_report_bundle(
    report: BaseModel,
    output_dir: str | Path,
    *,
    stem: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Save a reproducible report bundle.

    Files:
    - ``<stem>.json``: complete structured report
    - ``<stem>_summary.csv``: aggregate metrics
    - ``<stem>_cases.csv``: per-case results
    - ``<stem>_summary.txt``: human-readable summary
    """

    if not stem.strip():
        raise ValueError(
            "Report stem cannot be empty."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "json": save_report_json(
            report,
            output_dir
            / f"{stem}.json",
            metadata=metadata,
        ),
        "summary_csv": (
            save_report_summary_csv(
                report,
                output_dir
                / f"{stem}_summary.csv",
                metadata=metadata,
            )
        ),
        "cases_csv": save_report_cases_csv(
            report,
            output_dir
            / f"{stem}_cases.csv",
        ),
        "summary_text": (
            save_report_summary_text(
                report,
                output_dir
                / f"{stem}_summary.txt",
                metadata=metadata,
            )
        ),
    }
