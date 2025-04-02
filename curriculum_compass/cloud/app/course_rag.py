import logging
from course_search import CloudCourseSearchSystem
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudCourseRAGPipeline:
    """
    TF-IDF based RAG pipeline for course information
    """
    
    def __init__(self, reranker):
        """
        Initialize the course RAG pipeline
        
        Args:
            reranker: Reranker instance
        """
        self.course_search_system = CloudCourseSearchSystem()
        self.reranker = reranker
        self.config = get_config()
        
        # Initialize by loading course data
        self.course_search_system.add_course_sentences_to_db()
    
    def retrieve_courses(self, query, top_k=10):
        """
        Retrieve relevant course information
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            
        Returns:
            List of course documents
        """
        results = self.course_search_system.query_courses(query, top_k)
        # Flatten the nested list structure
        return [doc for sublist in results["documents"] for doc in sublist]
    
    def rerank_courses(self, query, retrieved_docs, top_k=5):
        """
        Cross-encoder reranking of retrieved documents
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved documents
            top_k: Number of results to return
            
        Returns:
            List of reranked documents
        """
        try:
            if not retrieved_docs:
                logger.warning("No documents to rerank")
                return []
                
            if len(retrieved_docs) < top_k:
                logger.warning(f"Requested top_k={top_k} but only {len(retrieved_docs)} documents available")
                top_k = len(retrieved_docs)
                
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)
            logger.info(f"Successfully reranked {len(reranked_docs)} course documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during course reranking: {str(e)}")
            # Fall back to original documents if reranking fails
            return retrieved_docs[:top_k]
    
    def __call__(self, query, initial_k=10, final_k=5):
        """
        Process a query through the course RAG pipeline
        
        Args:
            query: User query string
            initial_k: Initial number of results to retrieve
            final_k: Final number of results after reranking
            
        Returns:
            List of reranked course documents
        """
        logger.info(f"Processing course query: {query}")
        
        logger.info("Retrieving relevant course information...")
        retrieved_docs = self.retrieve_courses(query, top_k=initial_k)

        logger.info("Cross-encoder reranking course information...")
        reranked_docs = self.rerank_courses(query, retrieved_docs, top_k=final_k)
        
        return reranked_docs