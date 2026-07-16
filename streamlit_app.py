import streamlit as st
from models.models import embeddings_model, google_model, query_checker
from langchain_postgres import PGVector
from dotenv import load_dotenv
import json
import os
from prompt.prompts import simple_prompt, default_prompt

load_dotenv()

st.set_page_config(
    page_title="LOTR RAG Assistant",
    page_icon="🧝‍♂️",
    layout="centered"
)

st.title("Lord of the Rings RAG Chat")
st.subheader("Retrieval-Augmented Generation\nQuery Router • Token Streaming • PGVector + Google Gemini")

@st.cache_resource(show_spinner="Connecting to PostgreSQL + PGVector...")
def get_vectorstore():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("DATABASE_URL not found in environment variables!")
        st.stop()
    db = PGVector(embeddings=embeddings_model, connection=database_url)
    return db

@st.cache_resource
def get_retriever(_db, k: int = 8):
    return db.as_retriever(search_kwargs={"k": k})

db = get_vectorstore()

# ─────────────────────────────────────────────────────────────
# Core Logic (preserved from your original code)
# ─────────────────────────────────────────────────────────────
def check_query(query: str) -> bool:
    """Your original query_checker logic."""
    response = query_checker.invoke(query)
    return getattr(response, "is_about_lotr", False)

def save_retrieved_documents(documents, filename: str = "retrieved_chunks.json"):
    """Your original save_json behavior."""
    try:
        docs_as_dict = [doc.model_dump(mode="json") for doc in documents]
        with open(filename, "w", encoding="utf-8") as j:
            json.dump(docs_as_dict, j, ensure_ascii=False, indent=2)
        return filename
    except Exception as e:
        st.warning(f"Could not save retrieved documents: {e}")
        return None

def prepare_rag_context(documents) -> str:
    """Your original context cleaning logic."""
    relevant_data = [data.page_content for data in documents]
    stripped_data = [d.replace("\n\n", " ") for d in relevant_data]
    return "\n\n".join(stripped_data)

# ─────────────────────────────────────────────────────────────
# Streaming Helper (new)
# ─────────────────────────────────────────────────────────────
def stream_llm_response(chain, inputs: dict):
    """
    Generator that yields text chunks from the chain.
    Works with LCEL (prompt | model) and handles common chunk formats
    from Google Gemini / LangChain.
    """
    for chunk in chain.stream(inputs):
        # Most common case with modern LangChain + Gemini
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                # Handle possible structured content format
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if text:
                            yield text
                    elif isinstance(part, str):
                        yield part
            else:
                yield str(content)
        else:
            # Fallback for string chunks or other formats
            yield str(chunk)

# ─────────────────────────────────────────────────────────────
# Main Query Function (with streaming support)
# ─────────────────────────────────────────────────────────────
def run_query_with_streaming(query: str, db):
    """
    Orchestrates routing → retrieval (if needed) → streaming generation.
    Returns metadata + the final accumulated answer.
    """
    is_lotr = check_query(query)

    if is_lotr:
        retriever = get_retriever(db, k=8)
        documents = retriever.invoke(query)
        saved_file = save_retrieved_documents(documents)

        context = prepare_rag_context(documents)
        chain = simple_prompt | google_model
        inputs = {"context": context, "question": query}

        path = "RAG (LOTR knowledge base)"
        num_docs = len(documents)
        saved_path = saved_file
    else:
        chain = default_prompt | google_model
        inputs = {"question": query}
        path = "Direct LLM (general knowledge)"
        num_docs = 0
        saved_path = None

    # We return the generator + metadata. Streaming happens in the UI layer.
    return {
        "chain": chain,
        "inputs": inputs,
        "path": path,
        "is_lotr": is_lotr,
        "num_docs_retrieved": num_docs,
        "saved_json": saved_path,
    }

# ─────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    if db:
        st.success("✅ Vector database connected")
    else:
        st.error("❌ Database not connected")

    st.divider()

    show_debug = st.checkbox("Show retrieval debug info", value=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    if st.session_state.last_result and st.session_state.last_result.get("saved_json"):
        try:
            with open(st.session_state.last_result["saved_json"], "rb") as f:
                st.download_button(
                    label="📥 Download last retrieved chunks",
                    data=f,
                    file_name=st.session_state.last_result["saved_json"],
                    mime="application/json",
                    use_container_width=True
                )
        except Exception:
            pass

    st.caption(
    "Built by\n [Claude Daigan](https://lackey43.github.io/portfolio/)")
		
    # [GitHub Repository](https://github.com/yourusername/your-repo-name) •
    # [Live Demo on GitHub Pages](https://yourusername.github.io/your-repo-name)

# ─────────────────────────────────────────────────────────────
# Display Chat History
# ─────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────────────────────
# Chat Input + Streaming Response (main flow)
# ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about Lord of the Rings or general questions..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant response with streaming
    with st.chat_message("assistant"):
        # Phase 1: Routing + Retrieval (synchronous, shown with status)
        with st.status("Analyzing query and retrieving knowledge...", expanded=False) as status:
            result = run_query_with_streaming(prompt, db)
            st.session_state.last_result = result

            if result["is_lotr"]:
                status.update(
                    label=f"Retrieved {result['num_docs_retrieved']} documents • Generating answer...",
                    state="running"
                )
            else:
                status.update(label="Generating answer...", state="running")

        # Phase 2: Actual token streaming of the final LLM response
        message_placeholder = st.empty()
        full_response = ""

        try:
            # This is the key streaming part (similar to your original OpenAI reference)
            stream = stream_llm_response(result["chain"], result["inputs"])

            # Use Streamlit's built-in write_stream for clean token rendering
            # (or we can do manual accumulation if you prefer more control)
            streamed_text = st.write_stream(stream)

            # st.write_stream returns the full concatenated string
            full_response = streamed_text if streamed_text else ""

            # Fallback: if write_stream didn't work well with your chain format, do manual streaming
            if not full_response:
                for token in stream_llm_response(result["chain"], result["inputs"]):
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error during generation: {str(e)}"
            message_placeholder.error(full_response)

        # Show debug info after streaming finishes
        if show_debug and result:
            with st.expander("🔍 Query Routing & Retrieval Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Query Type", "LOTR-related" if result["is_lotr"] else "General")
                    st.write(f"**Path taken:** {result['path']}")
                with col2:
                    if result["is_lotr"]:
                        st.metric("Documents Retrieved", result["num_docs_retrieved"])
                        if result.get("saved_json"):
                            st.caption(f"Saved to: `{result['saved_json']}`")
                    else:
                        st.caption("No retrieval performed")

                # NEW: Show the actual retrieved documents as JSON
                if result.get("is_lotr") and result.get("saved_json"):
                    try:
                        with open(result["saved_json"], "r", encoding="utf-8") as f:
                            retrieved_docs = json.load(f)
                        
                        with st.expander("📄 View Retrieved Documents (JSON)", expanded=False):
                            st.json(retrieved_docs, expanded=True)
                            st.caption(f"Showing {len(retrieved_docs)} document chunks from PGVector")
                    except Exception as e:
                        st.caption(f"Could not load retrieved JSON: {e}")

    # 3. Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ How this RAG Chatbot Works"):
    st.markdown("""
    **Smart Routing for Accurate Answers**

    *Example:*  'What is the story of Bilbo Baggins'

    This chatbot uses intelligent routing to deliver better responses:

    - **LOTR-related questions**: Retrieves the top 8 relevant chunks using vector stores from the Postgresql database, saves them as JSON, and generates grounded answers.
    - **General questions**: Responds directly with the default model (no retrieval).

    You get real-time streaming + full transparency in the debug panel.

    **Business Uses**  
    Perfect for customer support, internal knowledge bases, employee training, document Q&A, and any domain-specific chatbot that needs accurate, source-grounded responses.

    ---
    **Built by Claude Daigan**
    """)

st.caption(
    "Built by [Claude Daigan](https://www.linkedin.com/in/claude-lester-d-a84558288/) • "
    "[GitHub Repository](https://github.com/Lackey43/Chatbot-with-RAG) • "
    "[Portfolio](https://lackey43.github.io/portfolio/)"
)