import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

OUTPUT_DIR = RESULTS_DIR / "phase22g_final_analysis"

INPUT_FILES = {
    "chunking": (RESULTS_DIR / "phase22b_chunking_comparison.json"),
    "retrieval": (RESULTS_DIR / "phase22c_retrieval_comparison.json"),
    "embedding": (RESULTS_DIR / "phase22d_embedding_comparison.json"),
    "topk": (RESULTS_DIR / "phase22e_topk_comparison.json"),
    "threshold": (RESULTS_DIR / "phase22f_abstention_threshold_comparison.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one required experiment JSON artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 22 result not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")

    return data


def as_float(value: Any) -> float | None:
    """Convert numeric JSON values to float."""

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)

        if math.isfinite(number):
            return number

    return None


def fmt(value: Any, digits: int = 4) -> str:
    """Format report values consistently."""

    number = as_float(value)

    if number is None:
        return ""

    return f"{number:.{digits}f}"


def find_result(results: list[dict[str, Any]], *, key: str, value: Any) -> dict[str, Any]:
    """Find one unique experiment result."""

    matches = [item for item in results if item.get(key) == value]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one result where {key}={value!r}; found {len(matches)}."
        )

    return matches[0]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a compact Markdown table."""

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


def escape_xml(value: str) -> str:
    """Escape minimal XML entities."""

    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def svg_bar_chart(
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> str:
    """
    Create a dependency-free SVG grouped bar chart.

    The chart intentionally uses currentColor and opacity rather than
    hard-coded colors so it inherits the viewer's text color.
    """

    width = 1000
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 140

    plot_width = width - margin_left - margin_right

    plot_height = height - margin_top - margin_bottom

    if not categories:
        raise ValueError("SVG chart requires at least one category.")

    if not series:
        raise ValueError("SVG chart requires at least one series.")

    category_width = plot_width / len(categories)

    bar_group_width = category_width * 0.72

    bar_width = bar_group_width / len(series)

    def y_position(value: float) -> float:
        clipped = min(y_max, max(y_min, value))

        fraction = (clipped - y_min) / (y_max - y_min)

        return margin_top + plot_height * (1.0 - fraction)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        ('<rect x="0" y="0" width="100%" height="100%" fill="none"/>'),
        (
            f'<text x="{width / 2:.1f}" y="36" '
            'text-anchor="middle" font-size="22" '
            'font-family="sans-serif" fill="currentColor">'
            f"{escape_xml(title)}</text>"
        ),
    ]

    for tick_index in range(6):
        fraction = tick_index / 5

        value = y_min + fraction * (y_max - y_min)

        y = y_position(value)

        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="currentColor" opacity="0.15"/>'
        )

        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" '
            'text-anchor="end" font-size="13" '
            'font-family="sans-serif" fill="currentColor">'
            f"{value:.2f}</text>"
        )

    for category_index, category in enumerate(categories):
        group_start = (
            margin_left + category_index * category_width + (category_width - bar_group_width) / 2
        )

        for series_index, (_series_name, values) in enumerate(series):
            if len(values) != len(categories):
                raise ValueError("Every SVG series must match category count.")

            value = values[category_index]

            x = group_start + series_index * bar_width

            y = y_position(value)

            bar_height = margin_top + plot_height - y

            opacity = 0.35 + (0.55 * (series_index + 1) / len(series))

            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" '
                f'width="{max(bar_width - 3, 1):.1f}" '
                f'height="{bar_height:.1f}" '
                'fill="currentColor" '
                f'opacity="{opacity:.2f}"/>'
            )

        center_x = margin_left + (category_index + 0.5) * category_width

        parts.append(
            f'<text x="{center_x:.1f}" '
            f'y="{margin_top + plot_height + 25}" '
            'text-anchor="middle" font-size="13" '
            'font-family="sans-serif" fill="currentColor">'
            f"{escape_xml(category)}</text>"
        )

    legend_y = height - 62

    legend_spacing = plot_width / len(series)

    for series_index, (series_name, _) in enumerate(series):
        x = margin_left + series_index * legend_spacing

        opacity = 0.35 + (0.55 * (series_index + 1) / len(series))

        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y - 12}" '
            'width="18" height="12" '
            'fill="currentColor" '
            f'opacity="{opacity:.2f}"/>'
        )

        parts.append(
            f'<text x="{x + 25:.1f}" y="{legend_y}" '
            'font-size="13" font-family="sans-serif" '
            'fill="currentColor">'
            f"{escape_xml(series_name)}</text>"
        )

    parts.append("</svg>")

    return "\n".join(parts)


def svg_line_chart(
    *,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    x_label: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> str:
    """Create a dependency-free SVG line chart."""

    width = 1000
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 120

    plot_width = width - margin_left - margin_right

    plot_height = height - margin_top - margin_bottom

    if len(x_values) < 2:
        raise ValueError("SVG line chart requires at least two x values.")

    x_min = min(x_values)

    x_max = max(x_values)

    if x_min == x_max:
        raise ValueError("SVG line chart x values must vary.")

    def x_position(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_position(value: float) -> float:
        clipped = min(y_max, max(y_min, value))

        return margin_top + (1.0 - ((clipped - y_min) / (y_max - y_min))) * plot_height

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        (
            f'<text x="{width / 2:.1f}" y="36" '
            'text-anchor="middle" font-size="22" '
            'font-family="sans-serif" fill="currentColor">'
            f"{escape_xml(title)}</text>"
        ),
    ]

    for tick_index in range(6):
        fraction = tick_index / 5

        value = y_min + fraction * (y_max - y_min)

        y = y_position(value)

        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="currentColor" opacity="0.15"/>'
        )

        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" '
            'text-anchor="end" font-size="13" '
            'font-family="sans-serif" fill="currentColor">'
            f"{value:.2f}</text>"
        )

    for value in x_values:
        x = x_position(value)

        parts.append(
            f'<text x="{x:.1f}" '
            f'y="{margin_top + plot_height + 26}" '
            'text-anchor="middle" font-size="13" '
            'font-family="sans-serif" fill="currentColor">'
            f"{value:g}</text>"
        )

    dash_patterns = ["", "8 5", "2 4", "12 4 2 4"]

    for series_index, (_series_name, values) in enumerate(series):
        if len(values) != len(x_values):
            raise ValueError("Every SVG series must match x-value count.")

        points = " ".join(
            (f"{x_position(x):.1f},{y_position(y):.1f}")
            for x, y in zip(x_values, values, strict=True)
        )

        dash = dash_patterns[series_index % len(dash_patterns)]

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""

        opacity = 0.55 + (0.35 * (series_index + 1) / len(series))

        parts.append(
            f'<polyline points="{points}" '
            'fill="none" stroke="currentColor" '
            'stroke-width="3" '
            f'opacity="{opacity:.2f}"'
            f"{dash_attr}/>"
        )

        for x, y in zip(x_values, values, strict=True):
            parts.append(
                f'<circle cx="{x_position(x):.1f}" '
                f'cy="{y_position(y):.1f}" r="4" '
                'fill="currentColor" '
                f'opacity="{opacity:.2f}"/>'
            )

    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 54}" '
        'text-anchor="middle" font-size="14" '
        'font-family="sans-serif" fill="currentColor">'
        f"{escape_xml(x_label)}</text>"
    )

    legend_y = height - 22

    legend_spacing = plot_width / len(series)

    for series_index, (series_name, _) in enumerate(series):
        x = margin_left + series_index * legend_spacing

        parts.append(
            f'<text x="{x:.1f}" y="{legend_y}" '
            'font-size="13" font-family="sans-serif" '
            'fill="currentColor">'
            f"{escape_xml(series_name)}</text>"
        )

    parts.append("</svg>")

    return "\n".join(parts)


def build_experiment_matrix(
    *, chunking: dict[str, Any], retrieval: dict[str, Any], embedding: dict[str, Any]
) -> list[dict[str, Any]]:
    """Create one normalized table for the main retrieval experiments."""

    rows: list[dict[str, Any]] = []

    experiment_specs = [
        ("chunking", chunking.get("results", []), "strategy"),
        ("retrieval", retrieval.get("results", []), "retriever"),
        ("embedding", embedding.get("results", []), "embedding_label"),
    ]

    for experiment, results, variant_key in experiment_specs:
        if not isinstance(results, list):
            raise ValueError(f"{experiment} results must be a list.")

        for item in results:
            if not isinstance(item, dict):
                continue

            rows.append(
                {
                    "experiment": experiment,
                    "variant": item.get(variant_key, ""),
                    "hit_at_1": item.get("hit_at_1"),
                    "hit_at_3": item.get("hit_at_3"),
                    "hit_at_5": item.get("hit_at_5"),
                    "mrr_at_5": item.get("mrr_at_5"),
                    "precision_at_k": item.get("precision_at_k"),
                    "recall_at_k": item.get("recall_at_k"),
                    "ndcg_at_k": item.get("ndcg_at_k"),
                    "page_recall_at_5": item.get("page_recall_at_5"),
                    "mean_query_retrieval_seconds": (item.get("mean_query_retrieval_seconds")),
                }
            )

    return rows


def build_error_analysis(
    *, topk: dict[str, Any], threshold: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Build case-level error/boundary analysis from available artifacts.

    Phase 22E provides case-level top-k retrieval metrics.
    Phase 22F provides case-level confidence scores and labels.
    """

    rows: list[dict[str, Any]] = []

    topk_cases = topk.get("cases", [])

    if isinstance(topk_cases, list):
        for item in topk_cases:
            if not isinstance(item, dict):
                continue

            if item.get("top_k") != 3:
                continue

            recall = as_float(item.get("recall_at_k"))

            page_recall = as_float(item.get("page_recall_at_k"))

            if (recall is not None and recall < 1.0) or (
                page_recall is not None and page_recall < 1.0
            ):
                rows.append(
                    {
                        "analysis_type": ("top_k_3_incomplete_retrieval"),
                        "case_id": item.get("case_id", ""),
                        "should_answer": True,
                        "value": recall,
                        "detail": (
                            "top_k=3 did not retain all labeled "
                            "relevant evidence; top_k=5 reached "
                            "full aggregate recall."
                        ),
                    }
                )

    threshold_cases = threshold.get("cases", [])

    scored = [
        item
        for item in threshold_cases
        if (isinstance(item, dict) and as_float(item.get("top_score")) is not None)
    ]

    answerable = [item for item in scored if item.get("should_answer") is True]

    unanswerable = [item for item in scored if item.get("should_answer") is False]

    if answerable:
        closest_answerable = min(answerable, key=lambda item: float(item["top_score"]))

        rows.append(
            {
                "analysis_type": ("confidence_boundary_answerable"),
                "case_id": closest_answerable.get("case_id", ""),
                "should_answer": True,
                "value": closest_answerable.get("top_score"),
                "detail": (
                    "Lowest answerable rank-1 cross-encoder "
                    "score; closest answerable case to the "
                    "production threshold."
                ),
            }
        )

    if unanswerable:
        closest_unanswerable = max(unanswerable, key=lambda item: float(item["top_score"]))

        rows.append(
            {
                "analysis_type": ("confidence_boundary_unanswerable"),
                "case_id": closest_unanswerable.get("case_id", ""),
                "should_answer": False,
                "value": closest_unanswerable.get("top_score"),
                "detail": (
                    "Highest unanswerable rank-1 cross-encoder "
                    "score; closest unanswerable case to the "
                    "production threshold."
                ),
            }
        )

    threshold_results = threshold.get("threshold_results", [])

    if isinstance(threshold_results, list):
        production = [
            item
            for item in threshold_results
            if (isinstance(item, dict) and as_float(item.get("threshold")) == 0.0)
        ]

        if len(production) == 1:
            row = production[0]

            false_answers = int(row.get("false_answers", 0))

            false_abstentions = int(row.get("false_abstentions", 0))

            if false_answers == 0 and false_abstentions == 0:
                rows.append(
                    {
                        "analysis_type": ("production_threshold_errors"),
                        "case_id": "",
                        "should_answer": "",
                        "value": 0,
                        "detail": (
                            "No answerability classification errors "
                            "were observed at threshold 0.0 on the "
                            "30-case evaluation set."
                        ),
                    }
                )

    return rows


def generate_charts(
    *,
    chunking: dict[str, Any],
    retrieval: dict[str, Any],
    embedding: dict[str, Any],
    topk: dict[str, Any],
    threshold: dict[str, Any],
) -> list[Path]:
    """Generate all Phase 22G SVG charts."""

    chart_dir = OUTPUT_DIR / "charts"

    chart_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    chunking_results = chunking["results"]

    chunking_categories = [str(item["strategy"]) for item in chunking_results]

    chunking_svg = svg_bar_chart(
        title=("Chunking Strategy Retrieval Quality"),
        categories=chunking_categories,
        series=[
            ("Hit@1", [float(item["hit_at_1"]) for item in chunking_results]),
            ("MRR@5", [float(item["mrr_at_5"]) for item in chunking_results]),
            ("Recall@5", [float(item["recall_at_k"]) for item in chunking_results]),
            ("nDCG@5", [float(item["ndcg_at_k"]) for item in chunking_results]),
        ],
    )

    path = chart_dir / "chunking_quality.svg"

    path.write_text(chunking_svg, encoding="utf-8")

    generated.append(path)

    retrieval_results = retrieval["results"]

    retrieval_svg = svg_bar_chart(
        title=("Retrieval Architecture Quality"),
        categories=[str(item["retriever"]) for item in retrieval_results],
        series=[
            ("Hit@1", [float(item["hit_at_1"]) for item in retrieval_results]),
            ("MRR@5", [float(item["mrr_at_5"]) for item in retrieval_results]),
            ("Recall@5", [float(item["recall_at_k"]) for item in retrieval_results]),
            ("nDCG@5", [float(item["ndcg_at_k"]) for item in retrieval_results]),
        ],
    )

    path = chart_dir / "retrieval_quality.svg"

    path.write_text(retrieval_svg, encoding="utf-8")

    generated.append(path)

    embedding_results = embedding["results"]

    embedding_svg = svg_bar_chart(
        title=("Embedding Model Retrieval Quality"),
        categories=[str(item["embedding_label"]) for item in embedding_results],
        series=[
            ("Hit@1", [float(item["hit_at_1"]) for item in embedding_results]),
            ("MRR@5", [float(item["mrr_at_5"]) for item in embedding_results]),
            ("Recall@5", [float(item["recall_at_k"]) for item in embedding_results]),
            ("nDCG@5", [float(item["ndcg_at_k"]) for item in embedding_results]),
        ],
    )

    path = chart_dir / "embedding_quality.svg"

    path.write_text(embedding_svg, encoding="utf-8")

    generated.append(path)

    topk_summaries = topk["summaries"]

    x_values = [float(item["top_k"]) for item in topk_summaries]

    topk_svg = svg_line_chart(
        title=("Final Context Top-K Trade-off"),
        x_values=x_values,
        x_label="Final top_k",
        series=[
            ("Precision@K", [float(item["mean_precision_at_k"]) for item in topk_summaries]),
            ("Recall@K", [float(item["mean_recall_at_k"]) for item in topk_summaries]),
            ("nDCG@K", [float(item["mean_ndcg_at_k"]) for item in topk_summaries]),
            ("PageRecall@K", [float(item["mean_page_recall_at_k"]) for item in topk_summaries]),
        ],
    )

    path = chart_dir / "topk_tradeoff.svg"

    path.write_text(topk_svg, encoding="utf-8")

    generated.append(path)

    threshold_results = threshold["threshold_results"]

    threshold_x = [float(item["threshold"]) for item in threshold_results]

    threshold_svg = svg_line_chart(
        title=("Abstention Threshold Calibration"),
        x_values=threshold_x,
        x_label=("Minimum rank-1 cross-encoder score"),
        series=[
            ("Accuracy", [float(item["accuracy"]) for item in threshold_results]),
            ("Answer F1", [float(item["answer_f1"]) for item in threshold_results]),
            ("Abstention F1", [float(item["abstention_f1"]) for item in threshold_results]),
        ],
    )

    path = chart_dir / "threshold_calibration.svg"

    path.write_text(threshold_svg, encoding="utf-8")

    generated.append(path)

    return generated


def build_report(
    *,
    chunking: dict[str, Any],
    retrieval: dict[str, Any],
    embedding: dict[str, Any],
    topk: dict[str, Any],
    threshold: dict[str, Any],
    error_rows: list[dict[str, Any]],
) -> str:
    """Build the final Markdown experiment analysis."""

    chunking_results = chunking["results"]

    retrieval_results = retrieval["results"]

    embedding_results = embedding["results"]

    topk_results = topk["summaries"]

    threshold_results = threshold["threshold_results"]

    fixed = find_result(chunking_results, key="strategy", value="fixed")

    semantic = find_result(chunking_results, key="strategy", value="semantic")

    dense = find_result(retrieval_results, key="retriever", value="dense")

    reranked = find_result(retrieval_results, key="retriever", value="rrf_reranker")

    baseline_embedding = find_result(
        embedding_results, key="embedding_label", value="all_minilm_l6_v2"
    )

    # Validates that the multi-QA run is present in the results.
    find_result(embedding_results, key="embedding_label", value="multi_qa_minilm_l6_cos_v1")

    mpnet_embedding = find_result(
        embedding_results, key="embedding_label", value="all_mpnet_base_v2"
    )

    topk_3 = find_result(topk_results, key="top_k", value=3)

    topk_5 = find_result(topk_results, key="top_k", value=5)

    topk_10 = find_result(topk_results, key="top_k", value=10)

    topk_20 = find_result(topk_results, key="top_k", value=20)

    threshold_0 = find_result(threshold_results, key="threshold", value=0.0)

    score_distribution = threshold["score_distribution"]

    separation_gap = score_distribution.get("separation_gap")

    chunking_table = markdown_table(
        [
            "Strategy",
            "Chunks",
            "Avg words",
            "Hit@1",
            "MRR@5",
            "Precision@5",
            "Recall@5",
            "nDCG@5",
            "Page Recall@5",
            "Mean latency (s)",
        ],
        [
            [
                str(item["strategy"]),
                str(item["chunk_count"]),
                fmt(item["average_chunk_words"], 2),
                fmt(item["hit_at_1"]),
                fmt(item["mrr_at_5"]),
                fmt(item["precision_at_k"]),
                fmt(item["recall_at_k"]),
                fmt(item["ndcg_at_k"]),
                fmt(item["page_recall_at_5"]),
                fmt(item["mean_query_retrieval_seconds"]),
            ]
            for item in chunking_results
        ],
    )

    retrieval_table = markdown_table(
        [
            "Retriever",
            "Hit@1",
            "MRR@5",
            "Precision@5",
            "Recall@5",
            "nDCG@5",
            "Page Recall@5",
            "Mean latency (s)",
        ],
        [
            [
                str(item["retriever"]),
                fmt(item["hit_at_1"]),
                fmt(item["mrr_at_5"]),
                fmt(item["precision_at_k"]),
                fmt(item["recall_at_k"]),
                fmt(item["ndcg_at_k"]),
                fmt(item["page_recall_at_5"]),
                fmt(item["mean_query_retrieval_seconds"]),
            ]
            for item in retrieval_results
        ],
    )

    embedding_table = markdown_table(
        [
            "Embedding",
            "Dim",
            "Hit@1",
            "MRR@5",
            "Recall@5",
            "nDCG@5",
            "Page Recall@5",
            "Indexing (s)",
        ],
        [
            [
                str(item["embedding_label"]),
                str(item["embedding_dimension"]),
                fmt(item["hit_at_1"]),
                fmt(item["mrr_at_5"]),
                fmt(item["recall_at_k"]),
                fmt(item["ndcg_at_k"]),
                fmt(item["page_recall_at_5"]),
                fmt(item["indexing_seconds"]),
            ]
            for item in embedding_results
        ],
    )

    topk_table = markdown_table(
        [
            "Top-K",
            "Hit@K",
            "MRR@K",
            "Precision@K",
            "Recall@K",
            "nDCG@K",
            "Page Recall@K",
            "Mean latency (s)",
        ],
        [
            [
                str(item["top_k"]),
                fmt(item["mean_hit_at_k"]),
                fmt(item["mean_reciprocal_rank_at_k"]),
                fmt(item["mean_precision_at_k"]),
                fmt(item["mean_recall_at_k"]),
                fmt(item["mean_ndcg_at_k"]),
                fmt(item["mean_page_recall_at_k"]),
                fmt(item["mean_query_retrieval_seconds"]),
            ]
            for item in topk_results
        ],
    )

    threshold_table = markdown_table(
        [
            "Threshold",
            "Accuracy",
            "Answer F1",
            "Abstention F1",
            "False answers",
            "False abstentions",
        ],
        [
            [
                fmt(item["threshold"], 1),
                fmt(item["accuracy"]),
                fmt(item["answer_f1"]),
                fmt(item["abstention_f1"]),
                str(item["false_answers"]),
                str(item["false_abstentions"]),
            ]
            for item in threshold_results
        ],
    )

    hit_delta = float(reranked["hit_at_1"]) - float(dense["hit_at_1"])

    mrr_delta = float(reranked["mrr_at_5"]) - float(dense["mrr_at_5"])

    recall_delta = float(reranked["recall_at_k"]) - float(dense["recall_at_k"])

    latency_delta = float(reranked["mean_query_retrieval_seconds"]) - float(
        dense["mean_query_retrieval_seconds"]
    )

    topk3_recall_gap = float(topk_5["mean_recall_at_k"]) - float(topk_3["mean_recall_at_k"])

    lines = [
        "# DocQ&A Phase 22G — Final Experiment Analysis",
        "",
        "## Final production configuration",
        "",
        "- Chunking: fixed 200 words with 40-word overlap.",
        "- Embedding: sentence-transformers/all-MiniLM-L6-v2 on CPU.",
        "- Retrieval: dense + BM25 -> RRF(k=60) -> CrossEncoder reranker.",
        "- Reranker candidate pool: 20.",
        "- Final context top_k: 5.",
        "- Answerability gate: rank-1 cross-encoder score with threshold 0.0.",
        "",
        "## 1. Chunking strategy comparison",
        "",
        chunking_table,
        "",
        (
            "Fixed 200/40 retained full Recall@5 and Page Recall@5 "
            f"({fmt(fixed['recall_at_k'])} / "
            f"{fmt(fixed['page_recall_at_5'])}) while also producing "
            f"Hit@1={fmt(fixed['hit_at_1'])} and "
            f"MRR@5={fmt(fixed['mrr_at_5'])}. "
            "Semantic chunking increased page-labeled precision but "
            f"reduced Recall@5 to {fmt(semantic['recall_at_k'])} "
            f"and Page Recall@5 to {fmt(semantic['page_recall_at_5'])}."
        ),
        "",
        "## 2. Retrieval architecture comparison",
        "",
        retrieval_table,
        "",
        (
            "Relative to dense-only retrieval, the full reranked stack "
            f"improved Hit@1 by {hit_delta:.4f}, MRR@5 by "
            f"{mrr_delta:.4f}, and Recall@5 by {recall_delta:.4f}. "
            "The cost was substantially higher retrieval latency: "
            f"+{latency_delta:.4f} seconds/query in this CPU benchmark."
        ),
        "",
        "## 3. Embedding model comparison",
        "",
        embedding_table,
        "",
        (
            "All three tested embedding models produced the same final "
            "retrieval-quality metrics under the reranked architecture. "
            "The production all-MiniLM-L6-v2 model therefore remains the "
            "lowest-change choice. all-mpnet-base-v2 doubled vector "
            f"dimension from {baseline_embedding['embedding_dimension']} "
            f"to {mpnet_embedding['embedding_dimension']} without a "
            "measured quality gain. The first-run model-load timings for "
            "new Hugging Face models include download overhead and should "
            "not be treated as steady-state load benchmarks."
        ),
        "",
        "## 4. Final context Top-K",
        "",
        topk_table,
        "",
        (
            f"Moving from top_k=3 to top_k=5 increased mean Recall@K by "
            f"{topk3_recall_gap:.4f} to "
            f"{fmt(topk_5['mean_recall_at_k'])}, and Page Recall@K reached "
            f"{fmt(topk_5['mean_page_recall_at_k'])}. "
            "Increasing context to 10 or 20 did not improve recall, while "
            f"precision fell to {fmt(topk_10['mean_precision_at_k'])} and "
            f"{fmt(topk_20['mean_precision_at_k'])}, respectively."
        ),
        "",
        "## 5. Abstention threshold calibration",
        "",
        threshold_table,
        "",
        (
            "At the production threshold 0.0, the 30-case evaluation set "
            f"had accuracy={fmt(threshold_0['accuracy'])}, "
            f"answer F1={fmt(threshold_0['answer_f1'])}, and "
            f"abstention F1={fmt(threshold_0['abstention_f1'])}, with "
            f"{threshold_0['false_answers']} false answers and "
            f"{threshold_0['false_abstentions']} false abstentions. "
            f"The observed answerable/unanswerable score separation gap "
            f"was {fmt(separation_gap)}."
        ),
        "",
        "## 6. Error and boundary analysis",
        "",
    ]

    if error_rows:
        error_table = markdown_table(
            ["Type", "Case", "Should answer", "Value", "Detail"],
            [
                [
                    str(row["analysis_type"]),
                    str(row["case_id"]),
                    str(row["should_answer"]),
                    fmt(row["value"]) if as_float(row["value"]) is not None else str(row["value"]),
                    str(row["detail"]),
                ]
                for row in error_rows
            ],
        )

        lines.extend([error_table, ""])

    lines.extend(
        [
            (
                "No answerability classification errors were observed at "
                "threshold 0.0 on this evaluation set. The useful remaining "
                "error analysis is therefore boundary-focused: identify "
                "the lowest-scoring answerable case, the highest-scoring "
                "unanswerable case, and any cases where top_k=3 failed to "
                "retain all labeled relevant evidence."
            ),
            "",
            "## 7. Baseline vs improved system",
            "",
            markdown_table(
                ["Metric", "Dense baseline", "Final reranked", "Delta"],
                [
                    [
                        "Hit@1",
                        fmt(dense["hit_at_1"]),
                        fmt(reranked["hit_at_1"]),
                        f"{hit_delta:+.4f}",
                    ],
                    [
                        "MRR@5",
                        fmt(dense["mrr_at_5"]),
                        fmt(reranked["mrr_at_5"]),
                        f"{mrr_delta:+.4f}",
                    ],
                    [
                        "Recall@5",
                        fmt(dense["recall_at_k"]),
                        fmt(reranked["recall_at_k"]),
                        f"{recall_delta:+.4f}",
                    ],
                    [
                        "Mean retrieval latency (s)",
                        fmt(dense["mean_query_retrieval_seconds"]),
                        fmt(reranked["mean_query_retrieval_seconds"]),
                        f"{latency_delta:+.4f}",
                    ],
                ],
            ),
            "",
            "## 8. Interpretation and limitations",
            "",
            (
                "- Retrieval conclusions are based on the current "
                "30-case labeled evaluation set (18 answerable, "
                "12 unanswerable) and the sample document."
            ),
            (
                "- Page labels are used when explicit chunk qrels are not "
                "available, so Precision/Recall/nDCG should be described as "
                "page-labeled retrieval metrics rather than exhaustive "
                "chunk-level judgments."
            ),
            (
                "- The perfect threshold-0 answerability result is strong "
                "evidence for this benchmark but should not be assumed to "
                "generalize to unseen document domains without further "
                "calibration."
            ),
            (
                "- Cross-encoder reranking provides the strongest measured "
                "retrieval completeness/ranking performance, but it is also "
                "the dominant retrieval-latency cost on CPU."
            ),
            (
                "- The local LLM-as-judge quality metrics from Phase 21 "
                "remain useful but are not ground-truth human judgments."
            ),
            "",
            "## 9. Phase 22 conclusion",
            "",
            (
                "The experiment matrix supports the existing production "
                "configuration: fixed 200/40 chunking, all-MiniLM-L6-v2 "
                "embeddings, hybrid dense+BM25 retrieval, RRF fusion, "
                "cross-encoder reranking, top_k=5, and confidence threshold "
                "0.0. The main measured trade-off is retrieval latency from "
                "the reranker in exchange for stronger first-result ranking "
                "and complete labeled recall."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Generate Phase 22G final tables, charts, and analysis."""

    print("Loading Phase 22 experiment artifacts...")

    data = {key: load_json(path) for key, path in INPUT_FILES.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_matrix = build_experiment_matrix(
        chunking=data["chunking"], retrieval=data["retrieval"], embedding=data["embedding"]
    )

    matrix_path = OUTPUT_DIR / "phase22g_experiment_matrix.csv"

    write_csv(matrix_path, experiment_matrix)

    error_rows = build_error_analysis(topk=data["topk"], threshold=data["threshold"])

    error_path = OUTPUT_DIR / "phase22g_error_analysis.csv"

    write_csv(error_path, error_rows)

    charts = generate_charts(
        chunking=data["chunking"],
        retrieval=data["retrieval"],
        embedding=data["embedding"],
        topk=data["topk"],
        threshold=data["threshold"],
    )

    report = build_report(
        chunking=data["chunking"],
        retrieval=data["retrieval"],
        embedding=data["embedding"],
        topk=data["topk"],
        threshold=data["threshold"],
        error_rows=error_rows,
    )

    report_path = OUTPUT_DIR / "phase22g_final_analysis.md"

    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "phase": "22G",
        "inputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in INPUT_FILES.items()},
        "outputs": {
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "experiment_matrix": str(matrix_path.relative_to(PROJECT_ROOT)),
            "error_analysis": str(error_path.relative_to(PROJECT_ROOT)),
            "charts": [str(path.relative_to(PROJECT_ROOT)) for path in charts],
        },
    }

    manifest_path = OUTPUT_DIR / "phase22g_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print("Phase 22G outputs:")
    print(f"  {report_path}")
    print(f"  {matrix_path}")
    print(f"  {error_path}")

    for chart in charts:
        print(f"  {chart}")

    print(f"  {manifest_path}")

    print()
    print("Phase 22G final analysis completed.")


if __name__ == "__main__":
    main()
