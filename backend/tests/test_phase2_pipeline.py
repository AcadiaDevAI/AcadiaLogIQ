from backend.services.contextual_ingestion_service import infer_document_type


def test_document_type_hint():
    assert infer_document_type("payments_runbook_v2.pdf", None) == "Runbook"