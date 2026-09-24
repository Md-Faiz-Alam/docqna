from typing import TypeAlias


MetadataFilterValue: TypeAlias = str | int
MetadataFilters: TypeAlias = dict[
    str,
    MetadataFilterValue,
]

SUPPORTED_METADATA_FILTER_FIELDS = frozenset(
    {
        "document_id",
        "session_id",
        "filename",
        "file_type",
        "document_version",
    }
)


def normalize_metadata_filters(
    metadata_filters: MetadataFilters | None,
) -> MetadataFilters | None:
    """
    Validate and normalize supported retrieval filters.

    Supported fields:
    - document_id
    - session_id
    - filename
    - file_type
    - document_version
    """

    if metadata_filters is None:
        return None

    normalized: MetadataFilters = {}

    for field_name, raw_value in metadata_filters.items():
        if (
            field_name
            not in SUPPORTED_METADATA_FILTER_FIELDS
        ):
            raise ValueError(
                "Unsupported metadata filter: "
                f"{field_name!r}."
            )

        if field_name == "document_version":
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value <= 0
            ):
                raise ValueError(
                    "document_version filter must be "
                    "a positive integer."
                )

            normalized[field_name] = raw_value
            continue

        if not isinstance(raw_value, str):
            raise ValueError(
                f"{field_name} filter must be a string."
            )

        value = raw_value.strip()

        if not value:
            raise ValueError(
                f"{field_name} filter cannot be empty."
            )

        if field_name == "file_type":
            value = value.lower().lstrip(".")

        normalized[field_name] = value

    return normalized or None
