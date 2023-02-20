"""Wrapper around FAISS vector database."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import uuid

import numpy as np
import pyarrow as pa

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


def dependable_lance_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import lance
    except ImportError:
        raise ValueError(
            "Could not import lance python package. "
            "Please it install it with `pip install pylance` "
        )
    return lance


class LanceDataset(VectorStore):
    """Wrapper around Lance dataset.

    To use, you should have the ``pylance`` python package installed.

    Example:
        .. code-block:: python

            from langchain import LanceDataset
            lance_dataset = LanceDataset(embedding_function, uri)
    """

    def __init__(
        self,
        embedding_function: Callable,
        uri: str
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        lance = dependable_lance_import()
        self.dataset = lance.dataset(uri)

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        raise NotImplementedError()

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        q = np.array(embedding, dtype=np.float32)
        tbl = self.dataset.to_table(columns=["document", "metadata"],
                                    nearest={"column": "vector", "q": q, "k": k, "nprobes": 10, "refine_factor": 10})
        documents = self._to_documents(tbl["document"].to_numpy(), tbl["metadata"].to_numpy())
        return list(zip(documents, tbl["score"].to_numpy().tolist()))

    @staticmethod
    def _to_documents(docarray, metaarray):
        for doc, meta in zip(docarray, metaarray):
            if meta is None:
                yield Document(page_content=doc)
            else:
                yield Document(page_content=doc, metadata=json.loads(meta))

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20
    ) -> List[Document]:
        raise NotImplementedError()

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        uri: str,
        metadatas: Optional[List[dict]] = None,
    ) -> LanceDataset:
        """Construct Lance wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Initializes the Lance dataset

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import lance_dataset
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                dataset = LanceDataset.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        ndims = len(embeddings[0])

        lance = dependable_lance_import()

        # TODO add a DocumentStore interface here that can
        #      write directly to Lance format
        schema = pa.schema([
            pa.field("id", pa.uint32(), False),
            pa.field("document", pa.utf8(), False),
            pa.field("metadata", pa.utf8(), True),
            pa.field("vector", pa.list_(pa.float32(), list_size=ndims), False)
        ])

        ids = pa.array(range(len(texts)), type=pa.float32())
        docs = pa.array(texts, type=pa.utf8())
        metadatas = metadatas if metadatas is not None else [None] * len(texts)
        meta = pa.array([json.dumps(m) if m is not None else None for m in metadatas])

        embeddings = np.array(embeddings, dtype=np.float32).ravel()
        vectors = pa.FixedSizeListArray.from_arrays(pa.array(embeddings, type=pa.float32()), list_size=ndims)
        tbl = pa.Table.from_arrays([ids, docs, meta, vectors], schema=schema)
        dataset = lance.write_dataset(tbl, uri)

        # TODO create index later
        #if len(ids) > 10000:
        #    num_partitions = np.log(len(ids))/np.log(2)
        #    dataset.create_index("vector",
        #                         index_type="IVF_PQ",
        #                         num_partitions=num_partitions,  # IVF
        #                         num_sub_vectors=16)  # PQ

        return cls(embedding.embed_query, uri)

    def save_local(self, uri: str) -> None:
        """Save Lance dataset to disk

        Args:
            uri: where to save the lance dataset
        """
        path = Path(uri)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        lance = dependable_lance_import()
        lance.write_dataset(self.dataset.to_table(), str(path))

    @classmethod
    def load_local(cls, uri: str, embeddings: Embeddings) -> LanceDataset:
        """Load FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            uri: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
        """
        path = Path(uri)
        return cls(embeddings.embed_query, str(path))


import lance
import numpy as np
import pandas as pd
import pyarrow as pa

dd = {"tt123456": np.random.randn(96)}
df = pd.DataFrame({"vector": dd}).reset_index().rename(
    columns={"index": "content_id"})

schema = pa.schema([pa.field("content_id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=96))])

dataset = lance.write_dataset(df, "vectors.lance", schema=schema)
# dataset.create_index("vector", index_type="IVF_PQ", num_partitions=256)
dataset.to_table(nearest={"column": "vector",
                          "q": np.random.randn(96),
                          "k": 10}
                 ).to_pandas()