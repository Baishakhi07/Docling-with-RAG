from typing import List
from transformers import AutoTokenizer
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from lancedb.pydantic import LanceModel,vector
from langchain_ollama import OllamaLLM
import tiktoken  # For token counting



load_dotenv()

# --------------------------------------------------------------
# Local Model Setup
# --------------------------------------------------------------

# Initialize local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Initialize Ollama with Mistral
llm = OllamaLLM(model="mistral")


#tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)

# Local token counter
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

MAX_TOKENS = 8191  # Keep same chunk size limit
# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer = tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------

# Create a LanceDB database
db = lancedb.connect("data/lancedb")


# Define schema with local embeddings
class ChunkMetadata(BaseModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

class Chunks(LanceModel):
    text: str
    vector: list[float] = vector(dim=384)  # <-- use this helper here
    metadata: ChunkMetadata

# Create table
dummy_chunk = Chunks(
    text="dummy",
    vector=[0.0] * 384,
    metadata=ChunkMetadata(filename=None, page_numbers=None, title=None)
)
table = db.create_table("docling", data=[dummy_chunk], mode="overwrite")
table.delete("true")  # optional: clear the dummy row after schema is set



# --------------------------------------------------------------
# Process chunks with local embeddings
# --------------------------------------------------------------

processed_chunks = []
for chunk in chunks:
    # Get embedding as list (not numpy array)
    embedding = embedding_model.encode(chunk.text).tolist()

    # Create ChunkMetadata and Chunks instances
    '''chunk_meta = ChunkMetadata(
        filename=chunk.meta.origin.filename,
        page_numbers=[
            page_no
            for page_no in sorted(
                set(
                    prov.page_no
                    for item in chunk.meta.doc_items
                    for prov in item.prov
                )
            )
        ] or None,
        title=chunk.meta.headings[0] if chunk.meta.headings else None,
    )'''

    processed_chunks.append(
        Chunks(
            text=chunk.text,
            vector=embedding,
            metadata=ChunkMetadata(
                filename=chunk.meta.origin.filename,
                page_numbers=[
                    page_no
                    for page_no in sorted(
                        set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )
                    )
                ] or None,
                title=chunk.meta.headings[0] if chunk.meta.headings else None,
            )
        )
    )

# Add the validated chunks to the table
table.add(processed_chunks)

print(table.schema)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

#table.to_pandas()
#table.count_rows()
