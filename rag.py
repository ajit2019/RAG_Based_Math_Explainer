import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuration & Keys (validated on first use so Streamlit can hydrate secrets first)
load_dotenv()

# Model Constants
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.1-8b-instant"
VECTOR_STORE_PATH = Path(__file__).parent / "resources/math_vector_store"

llm = None
embeddings = None


def initialize_components():
    """Initializes the Embedding and LLM components."""
    global llm, embeddings
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "Missing API key. Set GROQ_API_KEY via .env, "
            "environment variables, or Streamlit secrets."
        )
    print("Initializing components...")
    if llm is None:
        llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.1,
            api_key=groq_key,
        )
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


_DEFAULT_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def process_urls(urls, embeddings_model=None):
    """Load pages from URLs, split into chunks, build FAISS index, persist, and return it.

    Fetches each URL separately so failures report the exact URL (e.g. connection aborted).
    """
    initialize_components()
    emb = embeddings_model if embeddings_model is not None else embeddings
    if not urls:
        raise ValueError("At least one URL is required.")

    documents = []
    failures: list[tuple[str, str]] = []
    for url in urls:
        u = url.strip()
        if not u:
            continue
        try:
            loader = WebBaseLoader(
                web_paths=[u],
                requests_kwargs={
                    "timeout": 60,
                    "headers": _DEFAULT_BROWSER_HEADERS,
                },
            )
            documents.extend(loader.load())
        except Exception as exc:
            failures.append((u, f"{type(exc).__name__}: {exc}"))

    if failures:
        detail = "\n".join(f"  • {u}\n    → {err}" for u, err in failures)
        if not documents:
            raise RuntimeError(
                "Could not load any URL. Fix or remove the failing link(s):\n" + detail
            ) from None
        print(
            "Warning: some URLs failed (others were indexed):\n" + detail,
            flush=True,
        )

    if not documents:
        raise ValueError("No page content was loaded; check your URLs.")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, emb)
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_PATH))
    return vector_store


def build_knowledge_base(urls, embeddings_model=None):
    """Build and persist the FAISS store from URLs."""
    return process_urls(urls, embeddings_model)


def initialize_math_rag():
    """Create shared LLM and embedding clients (for scripts or advanced use)."""
    initialize_components()
    return embeddings, llm


def get_math_help(vector_store, llm_client, student_query):
    """Retrieves context and generates an answer from the vector store.

    Uses langchain-core only (no langchain.chains), equivalent to stuff + retrieval.

    Returns:
        tuple[str, str]: (answer, newline-separated unique source hints from retrieved docs).
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are an Assistant for question-answer tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the question is in Hindi, answer in Hindi. If English, answer in English. "
        "Always show the formula used before solving. "
        "if you don't know the answer, say 'I don't know the answer to that question. Please try again with a different question.' "
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_store.as_retriever()
    print(f"\nThinking about: {student_query}")
    docs = retriever.invoke(student_query)
    context = format_docs(docs)
    chain = prompt | llm_client | StrOutputParser()
    answer = chain.invoke({"context": context, "input": student_query})
    sources_lines = []
    for d in docs:
        src = getattr(d, "metadata", None) and d.metadata.get("source")
        sources_lines.append(
            src if src else (d.page_content[:120] + "..." if d.page_content else "")
        )
    sources = "\n".join(dict.fromkeys(s for s in sources_lines if s))
    return answer, sources


def get_answer(vector_store, query):
    """Run RAG over `vector_store` for `query`; uses the shared LLM from initialize_components."""
    initialize_components()
    return get_math_help(vector_store, llm, query)


if __name__ == "__main__":
    math_urls = [
        "https://openstax.org/details/books/elementary-algebra-2e",
        "https://www.ck12.org/book/ck-12-algebra-i-second-edition/section/10.0/",
    ]

    v_store = process_urls(math_urls)
    query = "द्विघात समीकरण (Quadratic Equation) को हल करने का सूत्र क्या है?"
    answer, sources = get_answer(v_store, query)

    print("\n--- ASSISTANT RESPONSE ---")
    print(answer)
    if sources:
        print("\n--- SOURCES ---")
        print(sources)
