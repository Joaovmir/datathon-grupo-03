from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICIES_PATH = Path("data/policies/credit_policies.txt")
CHROMA_DIR = Path("artifacts/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index() -> Chroma:
    """
    Read the credit policy document, split into chunks, embed and persist
    to ChromaDB.

    Should be called once to generate the index. Subsequent calls overwrite
    the existing index.

    Returns:
        Chroma: Loaded vectorstore ready for search.
    """
    text = POLICIES_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.create_documents([text])

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=_embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def load_index() -> Chroma:
    """
    Load a previously built ChromaDB index from disk.

    Builds the index first if it does not exist yet.

    Returns:
        Chroma: Loaded vectorstore ready for search.
    """
    if not CHROMA_DIR.exists():
        return build_index()

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings,
    )


def search(query: str, vectorstore: Chroma, k: int = 3) -> str:
    """
    Retrieve the k most relevant policy chunks for a given query.

    Args:
        query: Natural language question about credit policies.
        vectorstore: Loaded Chroma vectorstore.
        k: Number of chunks to retrieve.

    Returns:
        Concatenated text of the most relevant policy chunks.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
