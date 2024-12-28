# Reranker Class
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Reranker:
    def __init__(self, reranker_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu"):
        """
        Initialize the Reranker with a cross-encoder model.
        Args:
            reranker_model_name: Name of the Hugging Face model for reranking.
            device: Device to run the model on ("cpu" or "cuda").
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name).to(device)

    def rerank(self, query, documents, top_k=None):
        """
        Rerank the documents based on their relevance to the query.
        Args:
            query: The input query string.
            documents: List of documents to rerank.
            top_k: Number of top documents to return (default: all).
        Returns:
            List of reranked documents.
        """
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
        ranked_documents = [documents[idx] for idx in ranked_indices]
        
        # Return the top_k documents if specified
        return ranked_documents[:top_k] if top_k else ranked_documents