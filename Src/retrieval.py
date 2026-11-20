from Src.domain_mapping import TYPE_TO_SEMANTIC_TAG

__all__ = ["TYPE_TO_SEMANTIC_TAG", "extract_context"]


def extract_context(start_index: int, end_index: int, document: str, range_size: int = 20) -> str:
    before_start = max(0, start_index - range_size)
    after_end = min(len(document), end_index + range_size)
    before = document[before_start:start_index]
    entity = document[start_index:end_index]
    after = document[end_index:after_end]
    return f"{before}**{entity}**{after}"
