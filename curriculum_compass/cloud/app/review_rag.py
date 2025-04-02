import logging
from review_search import GCPVectorSearchRetriever
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudReviewRAGPipeline:
    """
    Vector search based RAG pipeline for review information
    """
    
    def __init__(self, embedding_model, reranker, index_endpoint_id, deployed_index_id):
        """
        Initialize the review RAG pipeline
        
        Args:
            embedding_model: SentenceTransformer model
            reranker: Reranker instance
            index_endpoint_id: ID of the Vertex AI Vector Search endpoint
            deployed_index_id: ID of the deployed index
        """
        self.config = get_config()
        self.retriever = GCPVectorSearchRetriever(
            index_endpoint_id, 
            deployed_index_id,
            embedding_model_name=self.config.EMBEDDING_MODEL
        )
        self.reranker = reranker
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve relevant review documents
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            
        Returns:
            List of review documents
        """
        return self.retriever.retrieve(query, top_k=top_k)
    
    def rerank(self, query, retrieved_docs, top_k=5):
        """
        Rerank retrieved review documents
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved documents
            top_k: Number of results to return
            
        Returns:
            List of reranked documents
        """
        try:
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)
            logger.info(f"Successfully reranked {len(reranked_docs)} review documents")
            return reranked_docs
        except Exception as e:
            logger.error(f"Error during review reranking: {e}")
            return retrieved_docs[:top_k]
    
    def __call__(self, query, initial_k=10, final_k=5):
        """
        Process a query through the review RAG pipeline
        
        Args:
            query: User query string
            initial_k: Initial number of results to retrieve
            final_k: Final number of results after reranking
            
        Returns:
            List of reranked review documents
        """
        logger.info(f"Processing review query: {query}")
        
        logger.info("Retrieving relevant reviews...")
        retrieved_docs = self.retrieve(query, initial_k)

        logger.info("Reranking reviews...")
        reranked_docs = self.rerank(query, retrieved_docs, final_k)
        
        return reranked_docs