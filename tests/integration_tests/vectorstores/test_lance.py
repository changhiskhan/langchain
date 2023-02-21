"""Test Lance functionality."""
import tempfile

import pathlib
import pytest

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.wikipedia import Wikipedia
from langchain.vectorstores.lance_dataset import LanceDataset
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_lance(tmp_path: pathlib.Path) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = LanceDataset.from_texts(texts, FakeEmbeddings(), str(tmp_path / "dataset.lance"))
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_lance_local_save_load(tmp_path: pathlib.Path) -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = LanceDataset.from_texts(texts, FakeEmbeddings(), str(tmp_path / "dataset.lance"))

    new_dataset_uri = str(tmp_path / "new_dataset.lance")
    docsearch.save_local(new_dataset_uri)
    new_docsearch = LanceDataset.load_local(new_dataset_uri, FakeEmbeddings())
    output = new_docsearch.similarity_search("foo", k=1)
    expected = docsearch.similarity_search("foo", k=1)
    assert output == expected
