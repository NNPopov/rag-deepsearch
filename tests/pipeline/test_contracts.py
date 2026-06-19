"""R2 (red): ingest pipe contracts (Pydantic).

Pins the identity rules from Rag_pipeline_architecture.md §3:
  • Chunk.id == "{external_id}:{section_path}:{seq}" — deterministic business id;
  • content_hash == sha256(id + "\\x00" + text) — id mixed into the hash;
  • page_range is optional (None) — best-effort, not a boundary;
  • invalid input is rejected by validation.

All tests RED until S4: the ingest.contracts module does not exist yet.
"""
import hashlib

import pytest


def _chunk(**over):
    from ingest.contracts import Chunk

    kw = dict(
        external_id="test://ddia",
        section_path="1.2",
        chapter=1,
        heading="Leaders and Followers",
        seq=3,
        text="leader based replication",
        token_count=4,
    )
    kw.update(over)
    return Chunk(**kw)


def test_chunk_id_is_deterministic():
    c = _chunk()
    assert c.id == "test://ddia:1.2:3"


def test_content_hash_mixes_id_and_text():
    c = _chunk()
    expected = hashlib.sha256(
        (c.id + "\x00" + c.text).encode("utf-8")
    ).hexdigest()
    assert c.content_hash == expected


def test_same_text_different_section_differs():
    a = _chunk(section_path="1.2", seq=0)
    b = _chunk(section_path="3.1", seq=0)
    assert a.text == b.text
    assert a.content_hash != b.content_hash  # id mixed in → hashes diverge


def test_page_range_is_optional():
    c = _chunk(page_range=None)
    assert c.page_range is None
    c2 = _chunk(page_range=(24, 55))
    assert tuple(c2.page_range) == (24, 55)


def test_invalid_chunk_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _chunk(seq="not-an-int")
