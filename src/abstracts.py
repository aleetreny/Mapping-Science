from __future__ import annotations

from typing import Any


def reconstruct_abstract(abstract_inverted_index: Any) -> str | None:
    """Reconstruct an OpenAlex abstract from its inverted index."""
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return None

    positions: dict[int, str] = {}
    for word, indexes in abstract_inverted_index.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            return None
        for index in indexes:
            if not isinstance(index, int) or index < 0:
                return None
            if index in positions:
                return None
            positions[index] = word

    if not positions:
        return None

    max_position = max(positions)
    if len(positions) != max_position + 1:
        return None

    return " ".join(positions[index] for index in range(max_position + 1))
