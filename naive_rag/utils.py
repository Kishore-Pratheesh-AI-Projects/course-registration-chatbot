from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

def load_model_and_tokenizer(model_name: str):
    """Load a language model and its tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_embedding_model(model_name: str):
    """Load a SentenceTransformer embedding model.

    Args:
        model_name (str): The name of the embedding model.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    return SentenceTransformer(model_name)

def initialize_chromadb_client(db_path: str):
    """Initialize a ChromaDB Persistent Client.

    Args:
        db_path (str): Path to the ChromaDB persistent storage.

    Returns:
        chromadb.PersistentClient: Initialized ChromaDB client.
    """
    return chromadb.PersistentClient(path=db_path)
