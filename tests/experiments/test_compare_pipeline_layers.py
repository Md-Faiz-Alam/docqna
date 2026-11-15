from experiments.compare_pipeline_layers import LAYERS, render_table, summarize

# Four answerable and two unanswerable questions, as the LLM answered them.
RECORDS = [
    # answered correctly, strong retrieval, well grounded
    {
        "should_answer": True,
        "answered": True,
        "correct": True,
        "cited_relevant": True,
        "top_score": 4.0,
        "grounding_score": 8.0,
        "evidence_in_top_k": True,
        "seconds": 1.0,
    },
    # answered correctly, but weak retrieval score
    {
        "should_answer": True,
        "answered": True,
        "correct": True,
        "cited_relevant": True,
        "top_score": -11.0,
        "grounding_score": 6.0,
        "evidence_in_top_k": True,
        "seconds": 2.0,
    },
    # answered wrongly
    {
        "should_answer": True,
        "answered": True,
        "correct": False,
        "cited_relevant": False,
        "top_score": 1.0,
        "grounding_score": 0.5,
        "evidence_in_top_k": False,
        "seconds": 3.0,
    },
    # the model declined although the answer exists
    {
        "should_answer": True,
        "answered": False,
        "correct": False,
        "cited_relevant": False,
        "top_score": 2.0,
        "grounding_score": None,
        "evidence_in_top_k": True,
        "seconds": 1.5,
    },
    # made-up answer to an unanswerable question, off-topic retrieval
    {
        "should_answer": False,
        "answered": True,
        "correct": False,
        "cited_relevant": False,
        "top_score": -12.0,
        "grounding_score": 0.1,
        "evidence_in_top_k": None,
        "seconds": 1.0,
    },
    # made-up answer that passes the gate but is not grounded
    {
        "should_answer": False,
        "answered": True,
        "correct": False,
        "cited_relevant": False,
        "top_score": 0.5,
        "grounding_score": 1.0,
        "evidence_in_top_k": None,
        "seconds": 1.0,
    },
]


def test_without_safety_layers_every_answer_is_shown() -> None:
    row = summarize(RECORDS)

    assert (row["correct"], row["wrong"], row["declined_answerable"]) == (2, 1, 1)
    assert row["false_answers"] == 2
    assert row["evidence_in_top_k"] == 3
    assert row["median_seconds"] == 1.5


def test_gate_and_grounding_remove_made_up_answers_one_layer_at_a_time() -> None:
    gated = summarize(RECORDS, threshold=-10.5)
    grounded = summarize(RECORDS, threshold=-10.5, grounding_min=3.0)

    # The gate skips the off-topic question (and the weak-but-correct one).
    assert gated["false_answers"] == 1
    assert gated["correct"] == 1
    # The grounding check withholds the ungrounded made-up and wrong answers.
    assert grounded["false_answers"] == 0
    assert grounded["wrong"] == 0
    assert grounded["correct"] == 1


def test_table_lists_layers_in_order() -> None:
    row = summarize(RECORDS)
    result = {"provider": "test", "model": "m", "combined": dict.fromkeys(LAYERS, row)}

    lines = render_table(result).splitlines()
    assert [line.split("  ")[0].strip() for line in lines[3:]] == list(LAYERS.values())
