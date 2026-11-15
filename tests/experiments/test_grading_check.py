import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from experiments import grading_check
from experiments.grading_check import (
    PAGE_FIELDS,
    SAMPLE_FILE,
    compare,
    document_path,
    draw,
    kappa,
    load_items,
    make_handler,
    page_items,
)


def _row(group: str, number: int, auto: bool = True) -> dict[str, object]:
    return {
        "group": group,
        "case_id": f"{group}_{number:02d}",
        "should_answer": True,
        "question": "Q?",
        "answer": "A [1].",
        "reference": ["A"],
        "reference_kind": "expected",
        "context": "A.",
        "document": "doc.txt",
        "auto_correct": auto,
    }


def test_draw_is_reproducible_capped_and_numbered() -> None:
    groups = {"a": [_row("a", n) for n in range(30)], "b": [_row("b", n) for n in range(5)]}
    first = draw({k: [dict(r) for r in v] for k, v in groups.items()})
    second = draw({k: [dict(r) for r in v] for k, v in groups.items()})

    assert [i["case_id"] for i in first] == [i["case_id"] for i in second]
    assert sum(i["group"] == "a" for i in first) == 20
    assert sum(i["group"] == "b" for i in first) == 5
    assert [i["item_id"] for i in first] == [f"g{n:03d}" for n in range(1, 26)]


def test_page_never_sees_model_or_grade() -> None:
    item = {**_row("heldout/gemini", 1), "item_id": "g001"}
    (visible,) = page_items([item])

    assert set(visible) == set(PAGE_FIELDS)
    assert "auto_correct" not in visible and "group" not in visible and "case_id" not in visible


def test_kappa() -> None:
    assert kappa([(True, True), (False, False)]) == 1.0
    assert kappa([(True, False), (False, True)]) == -1.0
    assert kappa([(True, True), (True, True)]) is None
    assert kappa([]) is None


def test_compare_strict_and_lenient() -> None:
    items = [
        {**_row("g", 1, auto=True), "item_id": "g001"},
        {**_row("g", 2, auto=True), "item_id": "g002"},
        {**_row("g", 3, auto=False), "item_id": "g003"},
        {**_row("g", 4, auto=False), "item_id": "g004"},
        {**_row("g", 5, auto=True), "item_id": "g005"},
    ]
    labels = {
        "g001": {"label": "correct"},
        "g002": {"label": "partly"},
        "g003": {"label": "wrong"},
        "g004": {"label": "correct"},
        "g005": {"label": "unsure"},
    }
    strict = compare(items, labels, lenient=False)
    lenient = compare(items, labels, lenient=True)

    assert strict["compared"] == 4
    assert (strict["both_correct"], strict["auto_correct_human_not"]) == (1, 1)
    assert (strict["human_correct_auto_not"], strict["both_not_correct"]) == (1, 1)
    assert strict["agreement"] == 0.5
    assert (lenient["both_correct"], lenient["auto_correct_human_not"]) == (2, 0)
    assert lenient["agreement"] == 0.75


def test_documents_are_limited_to_the_sample() -> None:
    items = [{"document": "northwind_employee_handbook.txt"}]

    assert document_path("northwind_employee_handbook.txt", items) is not None
    assert document_path("../../.env", items) is None
    assert document_path("normans.txt", items) is None


def test_committed_sample_is_balanced_and_public_only() -> None:
    payload = json.loads(SAMPLE_FILE.read_text("utf-8"))
    items = payload["items"]
    groups = {i["group"] for i in items}

    assert len(items) == 80
    assert len({i["item_id"] for i in items}) == 80
    assert groups == {"public/gemini", "public/phi3_mini", "heldout/gemini", "heldout/phi3_mini"}
    assert all(sum(i["group"] == g for i in items) == 20 for g in groups)
    assert all(i["context"] and i["answer"] for i in items)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grading_check, "LABELS_FILE", tmp_path / "labels.json")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(load_items()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path / "labels.json"
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, body: dict[str, str]) -> int:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return error.code


def test_label_server_saves_labels_and_hides_grades(server) -> None:
    base, labels_file = server
    with urllib.request.urlopen(f"{base}/api/items") as response:
        data = json.loads(response.read())

    assert len(data["items"]) == 80 and data["labels"] == {}
    assert all(set(item) == set(PAGE_FIELDS) for item in data["items"])

    assert _post(f"{base}/api/label", {"item_id": "g001", "label": "correct", "note": "ok"}) == 200
    assert _post(f"{base}/api/label", {"item_id": "g001", "label": "maybe"}) == 400
    assert _post(f"{base}/api/label", {"item_id": "nope", "label": "wrong"}) == 400

    saved = json.loads(labels_file.read_text("utf-8"))["labels"]
    assert saved == {"g001": {"label": "correct", "note": "ok"}}
