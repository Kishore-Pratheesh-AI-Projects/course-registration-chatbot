from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def load_model_and_tokenizer(model_name: str):
    """Load a language model and its tokenizer."""
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

def clean_time(time_val):
    """Clean and format time values."""
    if pd.isna(time_val) or time_val == '' or time_val == 0:
        return None
    return str(int(time_val)).zfill(4)

def format_time(time_str):
    """Format time string to AM/PM format."""
    if not time_str or len(time_str) != 4:
        return None
    hours = int(time_str[:2])
    minutes = time_str[2:]
    period = "AM" if hours < 12 else "PM"
    if hours > 12:
        hours -= 12
    elif hours == 0:
        hours = 12
    return f"{hours}:{minutes} {period}"
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
