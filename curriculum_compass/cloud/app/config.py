import os
from dotenv import load_dotenv

# Load environment variables from .env file (useful for local development)
load_dotenv()

# Application configuration
class Config:
    # GCP Project configuration
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    REGION = os.getenv("GCP_REGION", "us-central1")
    
    # Storage configuration
    RAW_DATA_BUCKET = os.getenv("RAW_DATA_BUCKET", "curriculum-compass-raw-data")
    PROCESSED_DATA_BUCKET = os.getenv("PROCESSED_DATA_BUCKET", "curriculum-compass-processed-data")
    EMBEDDINGS_BUCKET = os.getenv("EMBEDDINGS_BUCKET", "curriculum-compass-embeddings")
    LOGS_BUCKET = os.getenv("LOGS_BUCKET", "curriculum-compass-logs")
    
    # Vector search configuration
    COURSE_INDEX_ENDPOINT = os.getenv("COURSE_INDEX_ENDPOINT")
    COURSE_DEPLOYED_INDEX = os.getenv("COURSE_DEPLOYED_INDEX", "curriculum-compass-course-index")
    REVIEW_INDEX_ENDPOINT = os.getenv("REVIEW_INDEX_ENDPOINT")
    REVIEW_DEPLOYED_INDEX = os.getenv("REVIEW_DEPLOYED_INDEX", "curriculum-compass-review-index")
    
    # Model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
    
    # RAG pipeline configuration
    COURSE_K = int(os.getenv("COURSE_K", "15"))
    REVIEW_K = int(os.getenv("REVIEW_K", "15"))
    FINAL_K = int(os.getenv("FINAL_K", "10"))
    
    # System prompt (from your existing code)
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT"," ")

    
    # Content filtering - banned substrings from your config
    BANNED_SUBSTRINGS = os.getenv("BANNED_SUBSTRINGS", "").split(",") if os.getenv("BANNED_SUBSTRINGS") else []

# Function to get configuration
def get_config():
    return Config
