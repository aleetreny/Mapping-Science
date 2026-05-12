import pandas as pd

from src.download_state import completed_cell_keys_from_manifest


def test_completed_cell_keys_from_manifest() -> None:
    manifest = pd.DataFrame(
        [
            {"subfield_id": "1100", "publication_year": 2010, "status": "completed_target_met"},
            {"subfield_id": "1100", "publication_year": 2011, "status": "completed_shortfall"},
            {"subfield_id": "1100", "publication_year": 2012, "status": "failed"},
            {"subfield_id": "1100", "publication_year": 2013, "status": "rate_limited"},
        ]
    )

    completed = completed_cell_keys_from_manifest(manifest)

    assert ("1100", 2010) in completed
    assert ("1100", 2011) in completed
    assert ("1100", 2012) not in completed
    assert ("1100", 2013) not in completed
