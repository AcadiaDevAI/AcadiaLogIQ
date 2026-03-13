from backend.ingestion.structured_parser import extract_blocks_from_text, build_chunks


def test_structure_parser_detects_headings_and_bullets():
    text = """
    SYMPTOMS
    Service intermittently fails.

    - error 500
    - retry loop

    RESOLUTION
    Restart the worker.
    """
    blocks = extract_blocks_from_text(text)
    assert any(b.block_type == "heading" for b in blocks)
    assert any(b.block_type == "bullet" for b in blocks)

    chunks = build_chunks(blocks)
    assert len(chunks) >= 2
    assert any(c.section_heading == "SYMPTOMS" for c in chunks)