import streamlit as st
import lancedb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize models
@st.cache_resource
def init_models():
    """Initialize embedding model and LLM."""
    return {
        "embedding_model": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        "llm": Ollama(model="mistral")
    }

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection."""
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")

def get_context(query: str, table, embedding_model, num_results: int = 5) -> str:
    """Search the database for relevant context using local embeddings."""
    # Generate local embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Search with local embedding
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)

def get_chat_response(messages, context: str, llm) -> str:
    """Get streaming response from Ollama."""
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """
    
    # Format messages for Ollama
    prompt = f"{system_prompt}\n\n"
    for msg in messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += "Assistant:"
    
    # Stream the response
    response = st.write_stream(llm.stream(prompt))
    return response

# Initialize Streamlit app
st.title("📚 Document Q&A (Local)")

# Initialize models and database
models = init_models()
table = init_db()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, table, models["embedding_model"])
        st.write("Found relevant sections:")
        
        for chunk in context.split("\n\n"):
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")

            st.markdown(
                f"""
                <div style="margin:10px 0; padding:10px; border-radius:4px; background-color:#f0f2f6;">
                    <details>
                        <summary style="cursor:pointer; color:#0f52ba; font-weight:500;">{source}</summary>
                        <div style="font-size:0.9em; color:#666; font-style:italic;">Section: {title}</div>
                        <div style="margin-top:8px;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display assistant response
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context, models["llm"])

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})