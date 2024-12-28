import os
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import chromadb

from utils import load_embedding_model, initialize_chromadb_client

def load_reviews_data(file_path: Path) -> pd.DataFrame:
    """Load reviews data from a CSV file.

    Args:
        file_path (Path): Path to the CSV file containing reviews data.

    Returns:
        pd.DataFrame: DataFrame containing the reviews data.
    """
    return pd.read_csv(file_path)

def stringify_review_instance(row: pd.Series) -> str:
    """Convert a review row into a formatted string.

    Args:
        row (pd.Series): A row from the reviews DataFrame.

    Returns:
        str: A formatted string representing the review instance.
    """
    template = f"""Metadata:
    CRN: {row['CRN']}, Course Name: {row['Course Name']}, Instructor: {row['Instructor']},
    Course Number: {row['Subject']}{row['Course Number']}

    Question:
    {row['Question']}

    Review:
    {row['Review']}
    """
    return template

def prepare_corpus(data_frame: pd.DataFrame) -> list:
    """Convert a DataFrame of reviews into a list of formatted strings.

    Args:
        data_frame (pd.DataFrame): DataFrame containing the reviews data.

    Returns:
        list: A list of formatted strings representing the reviews.
    """
    return data_frame.apply(stringify_review_instance, axis=1).tolist()

def add_embeddings_to_collection(
    client: chromadb.PersistentClient, collection_name: str, texts: list, embeddings: np.ndarray
):
    """Add texts and embeddings to a ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): ChromaDB client instance.
        collection_name (str): Name of the ChromaDB collection.
        texts (list): List of texts to add to the collection.
        embeddings (np.ndarray): Numpy array of embeddings corresponding to the texts.

    Returns:
        None
    """
    collection = client.get_or_create_collection(collection_name)
    for idx, (text, embedding) in tqdm(enumerate(zip(texts, embeddings)), desc="Adding embeddings"):
        collection.add(
            documents=[text],
            metadatas=[{"index": idx}],
            ids=[str(idx)],
            embeddings=[embedding.tolist()]
        )
    print("All embeddings added to the collection.")

def embed_texts(texts: list, model_name: str) -> np.ndarray:
    """Generate embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts (list): List of texts to be embedded.
        model_name (str): Name of the SentenceTransformer model to use for embeddings.

    Returns:
        np.ndarray: Numpy array of embeddings for the input texts.
    """
    embedding_model = load_embedding_model(model_name)
    start_time = time()
    embeddings = embedding_model.encode(texts)
    end_time = time()
    print(f"Embedding completed in {end_time - start_time} seconds.")
    return embeddings


def main():
    # Define paths and parameters
    DATA_DIR = Path().cwd().parent / "data_pipeline" / "notebooks" / "data"
    REVIEWS_DATA_FILE = DATA_DIR / "reviews.csv"
    CHROMADB_PATH = "./chromadb"
    MODEL_NAME = 'all-MiniLM-L6-v2'
    COLLECTION_NAME = "naive_rag_embeddings"

    # Step 1: Load data
    reviews_df = load_reviews_data(REVIEWS_DATA_FILE)

    # Step 2: Prepare corpus
    stringified_reviews_list = prepare_corpus(reviews_df)

    # Step 3: Embed texts
    embeddings = embed_texts(stringified_reviews_list, MODEL_NAME)

    # Step 4: Initialize ChromaDB client
    client = initialize_chromadb_client(CHROMADB_PATH)

    # Step 5: Add embeddings to ChromaDB collection
    add_embeddings_to_collection(client, COLLECTION_NAME, stringified_reviews_list, embeddings)

if __name__ == "__main__":
    main()
