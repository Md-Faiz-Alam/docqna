import pytest

from experiments.compare_embedding_models import embedding_model_specs, safe_collection_name


def test_compare_embedding_models() -> None:
    specs = embedding_model_specs()

    assert [spec.label for spec in specs] == [
        "all_minilm_l6_v2",
        "multi_qa_minilm_l6_cos_v1",
        "all_mpnet_base_v2",
    ]

    assert specs[0].model_name == "sentence-transformers/all-MiniLM-L6-v2"

    assert specs[1].model_name == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    assert specs[2].model_name == "sentence-transformers/all-mpnet-base-v2"

    names = [safe_collection_name(spec.label) for spec in specs]

    assert len(names) == len(set(names))

    assert all(name.startswith("") for name in names)

    with pytest.raises(ValueError):
        safe_collection_name("   ")
