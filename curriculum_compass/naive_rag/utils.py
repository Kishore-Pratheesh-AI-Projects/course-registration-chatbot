from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """
    Load a language model and its tokenizer with proper device mapping handled by Accelerate
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"  # Let Accelerate handle device placement
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


def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

@weave.op(name="generate_llm_response")
def load_model_and_tokenizer(model_name: str):
    """
    Load a language model and its tokenizer with proper device mapping handled by Accelerate
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"  # Let Accelerate handle device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_llm_response(system_prompt: str, query: str, retrieved_docs: list, 
                         model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Generate response using the language model"""
    if retrieved_docs:
        context = "\n".join(retrieved_docs)
        user_content = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
    else:
        user_content = f"Query: {query}\n\nAnswer:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Don't manually specify device - use the device Accelerate chose
    model_inputs = tokenizer([text], return_tensors="pt")
    
    # Move inputs to the same device as the model
    if hasattr(model, 'device'):
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4098,
        temperature=0.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response
