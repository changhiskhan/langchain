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
        from lance.vector import vec_to_table
    except ImportError:
        raise ValueError(
            "Could not import lance python package. "
            "Please it install it with `pip install pylance` "
        )
    return lance, vec_to_table


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
        lance = dependable_lance_import()[0]
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
        with_index: bool = False,
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
        lance, vec_to_table = dependable_lance_import()

        tbl = vec_to_table(embeddings, check_ndim=False)

        docs = pa.array(texts, type=pa.utf8())
        tbl = tbl.append_column("document", docs)

        metadatas = metadatas if metadatas is not None else [None] * len(texts)
        meta = pa.array([json.dumps(m) if m is not None else None for m in metadatas])
        tbl = tbl.append_column("metadata", meta)

        dataset = lance.write_dataset(tbl, uri)

        ndim = tbl["vector"].type.list_size
        if with_index and len(tbl) > 10000 and ndim % 8 == 0:
            num_partitions = 2**(int(np.log(len(tbl)))-4)
            i = 1
            num_sub_vectors = ndim
            while num_sub_vectors > 64:
                num_sub_vectors = num_sub_vectors / (8 * i)
                i += 1
            dataset.create_index("vector",
                                 index_type="IVF_PQ",
                                 num_partitions=num_partitions,  # IVF
                                 num_sub_vectors=num_sub_vectors)  # PQ

        return cls(embedding.embed_query, uri)

    def save_local(self, uri: str) -> None:
        """Save Lance dataset to disk

        Args:
            uri: where to save the lance dataset
        """
        path = Path(uri)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        lance = dependable_lance_import()[0]
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
