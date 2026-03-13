from backend.services.contextual_ingestion_service import normalize_filename, parse_version_rank


def test_normalize_filename_removes_version_noise():
    assert normalize_filename("My SOP v2 FINAL.pdf") == "my-sop"


def test_parse_version_rank():
    assert parse_version_rank("v2.1") == 2.1
    assert parse_version_rank("Version 4") == 4.0
    assert parse_version_rank(None) == 0.0