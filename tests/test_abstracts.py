from src.abstracts import reconstruct_abstract


def test_reconstruct_abstract_from_inverted_index() -> None:
    inverted = {
        "OpenAlex": [0],
        "stores": [1],
        "abstracts": [2],
        "carefully": [3],
    }

    assert reconstruct_abstract(inverted) == "OpenAlex stores abstracts carefully"


def test_reconstruct_abstract_returns_none_for_missing_position() -> None:
    inverted = {"OpenAlex": [0], "abstracts": [2]}

    assert reconstruct_abstract(inverted) is None


def test_reconstruct_abstract_returns_none_for_empty_input() -> None:
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None
