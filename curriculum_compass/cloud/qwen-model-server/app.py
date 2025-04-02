import os
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Load model on startup
def load_model():
    global model, tokenizer
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto"  # Automatically use best available device
    )
    logger.info("Model loaded successfully")

# Run model loading on startup
load_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "Qwen2.5-3B-Instruct"})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get system prompt, query, and context
        system_prompt = data.get('system_prompt', '')
        query = data.get('query', '')
        context = data.get('context', '')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Format user content with context and query
        user_content = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
        
        # Create messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4098,
                temperature=0.1
            )
        
        # Process the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Generated response of length {len(response)}")
        
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)