import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudReranker:
    """
    Reranker implementation for GCP
    """
    
    def __init__(self, model_name):
        """
        Initialize the reranker
        
        Args:
            model_name: Name of the reranking model
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            logger.info(f"Initialized reranker with model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.tokenizer = None
            self.model = None
    
    def rerank(self, query, documents, top_k=None):
        """
        Rerank the documents based on their relevance to the query
        
        Args:
            query: The input query string
            documents: List of documents to rerank
            top_k: Number of top documents to return (default: all)
            
        Returns:
            List of reranked documents
        """
        if not self.model or not self.tokenizer or not documents:
            return documents[:top_k] if top_k else documents
            
        import torch
        
        try:
            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize inputs for the cross-encoder
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Predict relevance scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1)  # Extract scores from logits
            
            # Sort documents by scores in descending order
            ranked_indices = scores.argsort(descending=True)
            ranked_documents = [documents[idx.item()] for idx in ranked_indices]
            
            # Return the top_k documents if specified
            return ranked_documents[:top_k] if top_k else ranked_documents
        
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k] if top_k else documents