import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
# --------------------------------------------------------------
# Initialize local embedding model
# --------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dim


# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("docling")


# --------------------------------------------------------------
# Generate local embeddings for the query
# --------------------------------------------------------------
query = "what's docling?"
query_embedding = embedding_model.embed_query(query)  # Local embedding

# --------------------------------------------------------------
# Search the table with local embeddings
# --------------------------------------------------------------
results = table.search(query_embedding, vector_column_name="vector").limit(3) # Remove query_type="vector"
df = results.to_arrow().to_pandas()



