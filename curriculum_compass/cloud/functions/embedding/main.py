import functions_framework
import os
import json
import tempfile
from google.cloud import storage
from google.cloud import aiplatform
import traceback
from time import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embed_texts(texts, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for a list of texts using a SentenceTransformer model."""
    embedding_model = SentenceTransformer(model_name)
    start_time = time()
    embeddings = embedding_model.encode(texts)
    end_time = time()
    logger.info(f"Embedding completed in {end_time - start_time} seconds.")
    return embeddings

def create_vector_index_if_not_exists(index_id, embedding_dimension, region="us-central1"):
    """Create a Vector Search index if it doesn't already exist."""
    aiplatform.init(project=os.environ.get('PROJECT_ID'), location=region)
    
    # Check if index exists
    try:
        existing_indexes = aiplatform.MatchingEngineIndex.list()
        for idx in existing_indexes:
            if idx.display_name == index_id:
                logger.info(f"Vector index {index_id} already exists")
                return idx
    except Exception as e:
        logger.warning(f"Error checking existing indexes: {e}")
    
    # Create new index
    try:
        logger.info(f"Creating new vector index: {index_id}")
        index = aiplatform.MatchingEngineIndex.create(
            display_name=index_id,
            dimensions=embedding_dimension,
            approximate_neighbors_count=20,
            distance_measure_type="DOT_PRODUCT_DISTANCE"
        )
        logger.info(f"Successfully created index: {index_id}")
        return index
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")
        raise

# def upload_vectors_to_index(index, vectors, documents, ids):
#     """Upload vectors to a Vector Search index."""
#     try:
#         # First, we deploy the index to an endpoint if not already deployed
#         endpoints = index.deployed_indexes
#         if not endpoints:
#             logger.info(f"Deploying index {index} to endpoint")
#             index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
#                 display_name=f"{index.display_name}-endpoint",
#                 public_endpoint_enabled=True
#             )
#             index_endpoint.deploy_index(
#                 index=index,
#                 deployed_index_id=index.display_name
#             )
#             logger.info(f"Index deployed to endpoint: {index_endpoint.resource_name}")
#         else:
#             logger.info(f"Index already deployed to endpoint")
#             index_endpoint = endpoints[0].index_endpoint
        
#         # Now, upload the vectors in batches
#         batch_size = 100  # Adjust based on your needs
#         total_batches = (len(vectors) + batch_size - 1) // batch_size
        
#         logger.info(f"Uploading {len(vectors)} vectors in {total_batches} batches")
        
#         for i in range(0, len(vectors), batch_size):
#             batch_end = min(i + batch_size, len(vectors))
#             batch_vectors = vectors[i:batch_end]
#             batch_docs = documents[i:batch_end]
#             batch_ids = ids[i:batch_end]
            
#             batch_items = []
#             for vec, doc, doc_id in zip(batch_vectors, batch_docs, batch_ids):
#                 batch_items.append({
#                     "id": doc_id,
#                     "embedding": vec.tolist(),
#                     "restricts": {"namespace": "course_data"},
#                     "metadata": {"document": doc}
#                 })
            
#             # Use the appropriate method to update the index
#             index_endpoint.upsert(
#                 deployed_index_id=index.display_name,
#                 items=batch_items
#             )
            
#             logger.info(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
        
#         logger.info(f"Successfully uploaded all vectors to index")
#         return True
    
#     except Exception as e:
#         logger.error(f"Error uploading vectors to index: {e}")
#         raise
def upload_vectors_to_index(index, deployed_index_id, vectors, documents, ids):
    """Upload vectors to a Vector Search index using the specified deployed index id."""
    try:
        # First, we deploy the index to an endpoint if not already deployed
        endpoints = index.deployed_indexes
        if not endpoints:
            logger.info(f"Deploying index {index.display_name} to endpoint")
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=f"{index.display_name}-endpoint",
                public_endpoint_enabled=True
            )
            index_endpoint.deploy_index(
                index=index,
                deployed_index_id=deployed_index_id  # Use the deployed_index_id from env
            )
            logger.info(f"Index deployed to endpoint: {index_endpoint.resource_name}")
        else:
            logger.info(f"Index already deployed to endpoint")
            index_endpoint = endpoints[0].index_endpoint
        
        # Now, upload the vectors in batches
        batch_size = 100  # Adjust based on your needs
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        logger.info(f"Uploading {len(vectors)} vectors in {total_batches} batches")
        
        for i in range(0, len(vectors), batch_size):
            batch_end = min(i + batch_size, len(vectors))
            batch_vectors = vectors[i:batch_end]
            batch_docs = documents[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            batch_items = []
            for vec, doc, doc_id in zip(batch_vectors, batch_docs, batch_ids):
                batch_items.append({
                    "id": doc_id,
                    "embedding": vec.tolist(),
                    "restricts": {"namespace": "course_data"},
                    "metadata": {"document": doc}
                })
            
            # Use the deployed_index_id from the environment variable
            index_endpoint.upsert(
                deployed_index_id=deployed_index_id,
                items=batch_items
            )
            
            logger.info(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
        
        logger.info(f"Successfully uploaded all vectors to index")
        return True
    
    except Exception as e:
        logger.error(f"Error uploading vectors to index: {e}")
        raise

@functions_framework.cloud_event
def generate_embeddings_function(cloud_event):
    """Cloud Function triggered by a new processed data file in Cloud Storage."""
    try:
        # Get storage client
        storage_client = storage.Client()
        
        # Get bucket and file info from event
        bucket_name = cloud_event.data["bucket"]
        file_name = cloud_event.data["name"]
        
        # Process only processed course files
        if not (file_name.startswith('processed/') and file_name.endswith('.jsonl')):
            logger.info(f"Skipping non-processed file: {file_name}")
            return {"success": True, "message": "Skipped non-processed file"}
        
        # Download file from GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Create a temporary file to download the data
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file_path = temp_file.name
        
        # Read JSONL file
        documents = []
        document_ids = []
        with open(temp_file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                documents.append(record["content"])
                document_ids.append(record["id"])
        
        os.unlink(temp_file_path)  # Clean up temp file
        
        # Generate embeddings
        model_name = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        logger.info(f"Generating embeddings for {len(documents)} documents using {model_name}")
        
        embeddings = embed_texts(documents, model_name)
        
        # Get embedding dimension
        embedding_dimension = embeddings[0].shape[0]
        logger.info(f"Embedding dimension: {embedding_dimension}")
        
        # Create or get vector index
        index_id = os.environ.get('REVIEW_INDEX_ENDPOINT', 'curriculum-compass-course-index')
        region = os.environ.get('GCP_REGION', 'us-central1')
        
        index = create_vector_index_if_not_exists(index_id, embedding_dimension, region)

        deployed_index_id = os.environ.get('REVIEW_DEPLOYED_INDEX', index.display_name)

        logger.info(f"Using deployed index ID: {deployed_index_id}")

        # Pass both the index object and the deployed index id
        upload_result = upload_vectors_to_index(index, deployed_index_id, embeddings, documents, document_ids)
        
        # Store embeddings in Cloud Storage as backup
        output_bucket_name = os.environ.get('EMBEDDINGS_BUCKET', 'curriculum-compass-embeddings')
        output_bucket = storage_client.bucket(output_bucket_name)
        
        # Save embeddings as numpy file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_np:
            np.savez_compressed(
                temp_np.name,
                embeddings=embeddings,
                document_ids=np.array(document_ids),
            )
            temp_np_path = temp_np.name
        
        # Upload to GCS
        base_name = os.path.basename(file_name)
        output_file_name = f"embeddings/{base_name.replace('_processed.jsonl', '')}_embeddings.npz"
        output_blob = output_bucket.blob(output_file_name)
        output_blob.upload_from_filename(temp_np_path)
        
        # Also save a "latest" version if this is the latest processed file
        if 'latest' in file_name:
            latest_output_name = "embeddings/courses_embeddings_latest.npz"
            latest_output_blob = output_bucket.blob(latest_output_name)
            latest_output_blob.upload_from_filename(temp_np_path)
        
        os.unlink(temp_np_path)  # Clean up temp file
        
        logger.info(f"Embeddings uploaded to gs://{output_bucket_name}/{output_file_name}")
        
        return {
            "success": True,
            "input_file": file_name,
            "vector_index": index_id,
            "embeddings_file": output_file_name,
            "document_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error in embedding generator function: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}