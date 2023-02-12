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
    docsearch = LanceDataset.from_texts(texts, FakeEmbeddings(), str(tmp_path))
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_lance_local_save_load(tmp_path: pathlib.Path) -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = LanceDataset.from_texts(texts, FakeEmbeddings(), str(tmp_path))

    with tempfile.NamedTemporaryFile() as temp_file:
        docsearch.save_local(temp_file.name)
        new_docsearch = LanceDataset.load_local(temp_file.name, FakeEmbeddings())
    output = new_docsearch.similarity_search("foo", k=1)
    expected = docsearch.similarity_search("foo", k=1)
    assert output == expected

