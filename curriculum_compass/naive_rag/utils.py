from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from data_processor import CourseDataProcessor
import json
from sentence_transformers import SentenceTransformer
import chromadb
import weave
import torch


def get_device():
    """
    Determines the best device to run the application based on hardware availability.
    Returns 'cuda' if CUDA is available, 'mps' if MPS is available, and 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

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



def load_course_data(file_path: str):
    """Load and process course data.
    Args:
        file_path (str): Path to the course data CSV file 
    Returns:
        list: Processed course data ready for RAG pipeline
    """
    try:
        return CourseDataProcessor.process_course_data(file_path)
    except Exception as e:
        print(f"Error loading course data: {str(e)}")
        return None



def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

@weave.op(name="generate_llm_response")
def generate_llm_response(system_prompt: str, query:str,retrieved_docs:list,model:AutoModelForCausalLM,tokenizer:AutoTokenizer):
    """Generate response using the language model"""
    # Join reranked documents into a context string
    context = "\n".join(retrieved_docs)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"}
    ]
    
    # Tokenize and generate
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4098,
        temperature=0.1
    )
    # Remove input tokens from output to isolate generated text
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
