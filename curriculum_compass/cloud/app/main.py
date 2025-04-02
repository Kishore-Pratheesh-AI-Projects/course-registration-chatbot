#TODO :  Add the Query validation using LLM Guard
import os
import json
import logging
import datetime
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from google.cloud import storage
from rag_service import CurriculumCompassRAG
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config = get_config()

# Initialize storage client
storage_client = storage.Client()

# Initialize RAG system
rag_system = CurriculumCompassRAG()

# Initalize the Query validation using LLM Guard

# Initialize Vertex AI for LLM
aiplatform.init(
    project=config.PROJECT_ID,
    location=config.REGION
)

# Set up LLM endpoint if configured
llm_endpoint = None
if config.LLM_ENDPOINT:
    try:
        llm_endpoint = aiplatform.Endpoint(config.LLM_ENDPOINT)
        logger.info(f"Connected to LLM endpoint: {config.LLM_ENDPOINT}")
    except Exception as e:
        logger.error(f"Error connecting to LLM endpoint: {e}")


def generate_response(query, context):
    #TODO : Check this function properly and update this code to to apply proper messgae template and format before passing it to LLM
    """Generate response using Vertex AI LLM endpoint"""
    if not llm_endpoint:
        return "LLM endpoint not configured. Please check your configuration."
    
    try:
        # Format prompt with system instruction and context
        system_prompt = config.SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
        
        # Prepare the request - format may vary depending on the specific model
        instances = [{
            "prompt": f"{system_prompt}\n\n{user_content}"
        }]
        
        # Call the model endpoint
        response = llm_endpoint.predict(instances=instances)
        
        # Extract the response text
        if hasattr(response, 'predictions') and response.predictions:
            return response.predictions[0]
        else:
            return "Unable to generate a response from the model."
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"


def log_conversation(session_id, query, response):
    """Log conversation to Cloud Storage for analytics"""
    try:
        # Create a log entry
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        
        # Save to Cloud Storage
        bucket_name = config.LOGS_BUCKET
        bucket = storage_client.bucket(bucket_name)
        
        # Create a unique filename
        filename = f"conversations/{session_id}_{int(datetime.datetime.now().timestamp())}.json"
        blob = bucket.blob(filename)
        
        # Upload the log
        blob.upload_from_string(json.dumps(log_entry))
        logger.info(f"Logged conversation to {filename}")
        
    except Exception as e:
        logger.error(f"Error logging conversation: {e}")


@app.route('/query', methods=['POST'])
def process_query():
    """Process a user query through the RAG pipeline"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query is required"}), 400
        
        query = data['query']
        session_id = data.get('session_id', 'anonymous')
        
        logger.info(f"Processing query from session {session_id}: {query}")

        # Validate query using LLM Guard (if implemented)
        
        # Retrieve and rerank documents
        documents = rag_system.retrieve_and_rerank(query)
        
        # If no documents were found
        if not documents:
            return jsonify({
                "status": "success",
                "response": "I don't have enough information to answer that question about Northeastern University courses.",
                "context_documents": 0
            })
        
        # Join documents into context
        context = "\n\n".join(documents)
        
        # Generate response
        response = generate_response(query, context)
        
        # Log conversation
        log_conversation(session_id, query, response)
        
        return jsonify({
            "status": "success",
            "response": response,
            "context_documents": len(documents)
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An error occurred while processing your query."
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    })


if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)