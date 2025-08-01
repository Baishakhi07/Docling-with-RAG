from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import tiktoken  # For token counting (local alternative to OpenAI tokenizer)

from transformers import AutoTokenizer


load_dotenv()

# --------------------------------------------------------------
# Local Model Setup (replaces OpenAI)
# --------------------------------------------------------------

# Initialize local embedding model (replace OpenAI embeddings)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Ollama with Mistral (replace OpenAI client)
llm = OllamaLLM(model="mistral")

# Local token counter (replace OpenAITokenizerWrapper)
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")  # Same tokenizer OpenAI uses
    return len(encoding.encode(text))

MAX_TOKENS = 8191  # Keep same chunk size limit

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

len(chunks)
