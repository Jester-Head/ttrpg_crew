"""Tests for when the knowledge index rebuilds (no chromadb or embedder needed)."""

import pytest

from ttrpg_crew.services import knowledge


class FakeCollection:
    def __init__(self, metadata=None):
        self.metadata = metadata
        self.ids: list[str] = []
        self.fail_after: int | None = None  # simulate an embedder dying mid-index

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids, documents):
        for chunk_id in ids:
            if self.fail_after is not None and len(self.ids) >= self.fail_after:
                raise RuntimeError("embedding server went away")
            self.ids.append(chunk_id)


class FakeClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}
        self.fail_after: int | None = None

    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        return self.collections.setdefault(name, FakeCollection(metadata))

    def create_collection(self, name, embedding_function=None, metadata=None):
        assert name not in self.collections, "create_collection on an existing name"
        col = self.collections[name] = FakeCollection(metadata)
        col.fail_after = self.fail_after
        return col

    def delete_collection(self, name):
        del self.collections[name]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    lore = tmp_path / "lore"
    lore.mkdir()
    (lore / "a.md").write_text("# Alpha\n\nFirst fact.\n", encoding="utf-8")
    (lore / "b.md").write_text("# Beta\n\nSecond fact.\n", encoding="utf-8")
    client = FakeClient()
    monkeypatch.setattr(knowledge, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(knowledge, "_RAG_GLOBS", ("lore/*.md",))
    monkeypatch.setattr(knowledge, "_client", lambda: client)
    monkeypatch.setattr(knowledge, "_embedding_fn", lambda: None)
    return lore, client


def _collection(client):
    return client.collections[knowledge._COLLECTION]


def test_first_run_indexes_and_stamps_fingerprint(corpus):
    _, client = corpus
    assert knowledge.ensure_indexed(verbose=False) == 2
    assert _collection(client).metadata["corpus_fingerprint"]


def test_unchanged_sources_reuse_the_index(corpus):
    _, client = corpus
    knowledge.ensure_indexed(verbose=False)
    first = _collection(client)
    assert knowledge.ensure_indexed(verbose=False) == 2
    assert _collection(client) is first  # not rebuilt


def test_edited_source_rebuilds(corpus):
    lore, client = corpus
    knowledge.ensure_indexed(verbose=False)
    first = _collection(client)
    (lore / "b.md").write_text("# Beta\n\nRevised fact.\n", encoding="utf-8")
    knowledge.ensure_indexed(verbose=False)
    assert _collection(client) is not first


def test_swapped_files_rebuild(corpus):
    lore, client = corpus
    knowledge.ensure_indexed(verbose=False)
    first = _collection(client)
    (lore / "b.md").unlink()
    (lore / "c.md").write_text("# Gamma\n\nNew fact.\n", encoding="utf-8")
    knowledge.ensure_indexed(verbose=False)
    assert _collection(client) is not first
    assert _collection(client).count() == 2


def test_legacy_index_without_fingerprint_rebuilds(corpus):
    _, client = corpus
    legacy = client.get_or_create_collection(knowledge._COLLECTION)
    legacy.ids = ["old-daggerheart-chunk"]
    knowledge.ensure_indexed(verbose=False)
    assert _collection(client) is not legacy
    assert "old-daggerheart-chunk" not in _collection(client).ids


def test_partial_index_from_a_failed_run_rebuilds(corpus):
    _, client = corpus
    client.fail_after = 1
    with pytest.raises(RuntimeError):
        knowledge.ensure_indexed(verbose=False)
    assert _collection(client).count() == 1  # stamped, but short
    client.fail_after = None
    assert knowledge.ensure_indexed(verbose=False) == 2


def test_embedding_model_change_rebuilds(corpus, monkeypatch):
    _, client = corpus
    knowledge.ensure_indexed(verbose=False)
    first = _collection(client)
    monkeypatch.setenv("LOCAL_EMBED_MODEL", "some-other-embedder")
    knowledge.ensure_indexed(verbose=False)
    assert _collection(client) is not first
