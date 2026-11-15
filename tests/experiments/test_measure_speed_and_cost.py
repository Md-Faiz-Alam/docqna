import pytest

from experiments.measure_speed_and_cost import PRICES, cost, percentile, summarize


def _answer(case_id: str, total: float, *, llm: bool = True) -> dict[str, object]:
    return {
        "case_id": case_id,
        "llm_calls": 1 if llm else 0,
        "answered": llm,
        "search_ms": 20.0,
        "rerank_ms": 100.0,
        "first_token_ms": total / 2 if llm else None,
        "generation_ms": total - 150 if llm else 0.0,
        "grounding_ms": 30.0 if llm else 0.0,
        "total_ms": total,
        "input_tokens": 1000 if llm else 0,
        "output_tokens": 40 if llm else 0,
        "thinking_tokens": 0,
    }


RUN = {
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite",
    "hardware": {},
    "startup_seconds": {},
    "indexing": [],
    "answers": [
        _answer("a", 1000.0),
        _answer("b", 2000.0),
        _answer("c", 3000.0),
        _answer("gated", 150.0, llm=False),
    ],
}


def test_percentile_picks_a_measured_value() -> None:
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.95) == 100.0
    assert percentile([], 0.5) is None


def test_cost_uses_published_prices_and_bills_thinking_as_output() -> None:
    price = PRICES["gemini-3.5-flash-lite"]
    usage = {"input_tokens": 1_000_000.0, "output_tokens": 500_000.0, "thinking_tokens": 500_000.0}

    assert cost(usage, "gemini-3.5-flash-lite") == pytest.approx(
        price["input_per_million"] + price["output_per_million"]
    )
    assert cost(usage, "phi3:mini") is None


def test_llm_stages_only_count_questions_that_reached_the_llm() -> None:
    summary = summarize(RUN)

    assert summary["questions"] == 4
    assert summary["questions_reaching_llm"] == 3
    assert summary["stages_ms"]["generation_ms"]["count"] == 3
    assert summary["stages_ms"]["total_ms"]["count"] == 4
    assert summary["stages_ms"]["first_token_ms"]["median"] == 1000.0
    assert summary["mean_tokens_per_question"]["input_tokens"] == 1000.0


def test_cost_per_1000_questions() -> None:
    price = PRICES["gemini-3.5-flash-lite"]
    expected = (1000 * price["input_per_million"] + 40 * price["output_per_million"]) / 1000

    assert summarize(RUN)["cost_usd"]["per_1000_questions"] == round(expected, 2)
