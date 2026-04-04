import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag import get_answer, process_urls


def _hydrate_env_from_streamlit_secrets() -> None:
    """Populate env vars from Streamlit secrets when running in cloud."""
    if "GOOGLE_API_KEY" not in os.environ and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


APP_TITLE = "Real Estate Research Tool"


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="wide")
    st.title(APP_TITLE)

    _hydrate_env_from_streamlit_secrets()
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
        st.error(
            "Missing API keys. Set `GOOGLE_API_KEY` and `GROQ_API_KEY` in your `.env`, "
            "environment, or Streamlit secrets."
        )
        st.stop()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("Knowledge base")
        url1 = st.text_input("URL 1")
        url2 = st.text_input("URL 2")
        url3 = st.text_input("URL 3")
        process_url_button = st.button("Process URLs", type="primary")

    if process_url_button:
        urls = [url for url in (url1, url2, url3) if url.strip()]
        if len(urls) == 0:
            st.warning("You must provide at least one valid url")
        else:
            try:
                with st.spinner("Loading and indexing pages..."):
                    st.session_state.vector_store = process_urls(urls)
            except RuntimeError as e:
                st.error(str(e))
            else:
                st.success("URLs processed. Ask a question below.")

    query = st.text_input("Question")
    if query:
        if st.session_state.vector_store is None:
            st.error("Process at least one URL first (sidebar).")
        else:
            try:
                answer, sources = get_answer(st.session_state.vector_store, query)
                st.header("Answer:")
                st.write(answer)

                if sources:
                    st.subheader("Sources:")
                    for source in sources.split("\n"):
                        st.write(source)
            except Exception as e:
                st.error(f"Could not get an answer: {e}")


if __name__ == "__main__":
    main()
