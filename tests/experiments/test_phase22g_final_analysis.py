from experiments.run_phase22g_final_analysis import (
    as_float,
    build_error_analysis,
    find_result,
    markdown_table,
    svg_bar_chart,
    svg_line_chart,
)


def test_phase22g_final_analysis() -> None:
    assert as_float(1) == 1.0

    assert as_float("1.0") is None

    results = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]

    assert find_result(results, key="name", value="b")["value"] == 2

    table = markdown_table(["A", "B"], [["1", "2"]])

    assert "| A | B |" in table
    assert "| 1 | 2 |" in table

    bar_svg = svg_bar_chart(title="Test", categories=["A", "B"], series=[("Metric", [0.5, 1.0])])

    assert "<svg" in bar_svg
    assert "Metric" in bar_svg

    line_svg = svg_line_chart(
        title="Test", x_values=[0.0, 1.0], x_label="X", series=[("Metric", [0.5, 1.0])]
    )

    assert "<svg" in line_svg
    assert "polyline" in line_svg

    topk = {
        "cases": [
            {"case_id": "case_1", "top_k": 3, "recall_at_k": 0.5, "page_recall_at_k": 0.5},
            {"case_id": "case_1", "top_k": 5, "recall_at_k": 1.0, "page_recall_at_k": 1.0},
        ]
    }

    threshold = {
        "cases": [
            {"case_id": "answerable_1", "should_answer": True, "top_score": 1.5},
            {"case_id": "unanswerable_1", "should_answer": False, "top_score": -2.0},
        ],
        "threshold_results": [{"threshold": 0.0, "false_answers": 0, "false_abstentions": 0}],
    }

    errors = build_error_analysis(topk=topk, threshold=threshold)

    analysis_types = {row["analysis_type"] for row in errors}

    assert "top_k_3_incomplete_retrieval" in analysis_types

    assert "confidence_boundary_answerable" in analysis_types

    assert "confidence_boundary_unanswerable" in analysis_types

    assert "production_threshold_errors" in analysis_types
