import logging
from google.cloud import aiplatform
from sentence_transformers import SentenceTransformer
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class GCPVectorSearchRetriever:
    """
    A class to handle retrieval from GCP Vertex AI Vector Search for reviews
    """
    
    def __init__(self, index_endpoint_id, deployed_index_id, embedding_model_name):
        """
        Initialize the Vector Search retriever
        
        Args:
            index_endpoint_id: ID of the Vector Search index endpoint
            deployed_index_id: ID of the deployed index
            embedding_model_name: Name of the embedding model to use
        """
        # Initialize Vertex AI
        self.config = get_config()
        aiplatform.init(
            project=self.config.PROJECT_ID,
            location=self.config.REGION
        )
        
        # Get the index endpoint
        try:
            self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_id)
            self.deployed_index_id = deployed_index_id
        except Exception as e:
            logger.error(f"Error initializing Vector Search endpoint: {e}")
            self.index_endpoint = None
            self.deployed_index_id = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Initialized GCP Vector Search retriever with model {embedding_model_name}")
    
    def retrieve(self, query, top_k=10, namespace="review_data"):
        """
        Retrieve documents based on query
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            namespace: Namespace to search in
            
        Returns:
            List of retrieved documents
        """
        if not self.index_endpoint:
            logger.warning("Vector Search endpoint not initialized, returning empty results")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search for similar vectors
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding.tolist()],
                num_neighbors=top_k
            )
            
            # Extract documents from the response
            documents = []
            for neighbor in response[0]:
                # The document content is stored in the metadata
                if hasattr(neighbor, 'metadata') and 'document' in neighbor.metadata:
                    documents.append(neighbor.metadata['document'])
                else:
                    # Fallback if metadata is not found
                    documents.append(f"Document ID: {neighbor.id}")
            
            logger.info(f"Retrieved {len(documents)} review documents for query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving from Vector Search: {e}")
            return []