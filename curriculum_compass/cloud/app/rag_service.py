import logging
from sentence_transformers import SentenceTransformer
from course_rag import CloudCourseRAGPipeline
from review_rag import CloudReviewRAGPipeline
from reranker import CloudReranker
from config import get_config


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumCompassRAG:
    """
    Main RAG implementation for Curriculum Compass that integrates
    course and review retrieval
    """
    
    def __init__(self):
        """
        Initialize the integrated RAG system
        """
        self.config = get_config()
        
        # Initialize reranker (used by both pipelines)
        self.reranker = CloudReranker(self.config.RERANKER_MODEL)
        
        # Initialize course RAG pipeline
        self.course_rag = CloudCourseRAGPipeline(self.reranker)
        
        # Initialize review RAG pipeline if endpoints are configured
        self.review_rag = None
        if self.config.REVIEW_INDEX_ENDPOINT:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.review_rag = CloudReviewRAGPipeline(
                embedding_model, 
                self.reranker,
                self.config.REVIEW_INDEX_ENDPOINT,
                self.config.REVIEW_DEPLOYED_INDEX
            )
        
        logger.info("Initialized Curriculum Compass RAG system")
    
    def retrieve_and_rerank(self, query):
        """
        Retrieve and rerank documents for a query
        
        Args:
            query: User query string
            
        Returns:
            List of reranked documents
        """
        # Retrieve course information
        course_docs = self.course_rag(
            query, 
            initial_k=self.config.COURSE_K, 
            final_k=self.config.COURSE_K
        )
        
        # Retrieve reviews if review_rag is initialized
        review_docs = []
        if self.review_rag:
            review_docs = self.review_rag(
                query, 
                initial_k=self.config.REVIEW_K, 
                final_k=self.config.REVIEW_K
            )
        
        # Combine documents with clear labeling
        combined_docs = []
        for doc in course_docs:
            combined_docs.append(f"[COURSE INFO] {doc}")
        for doc in review_docs:
            combined_docs.append(f"[STUDENT REVIEW] {doc}")
        
        # If no documents were retrieved, return empty list
        if not combined_docs:
            logger.warning(f"No documents retrieved for query: {query}")
            return []
        
        # Rerank combined documents
        reranked_docs = self.reranker.rerank(query, combined_docs, top_k=self.config.FINAL_K)
        logger.info(f"Retrieved and reranked {len(reranked_docs)} documents for query: {query}")
        
        return reranked_docs