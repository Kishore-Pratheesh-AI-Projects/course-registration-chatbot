import weave
from curriculum_compass.naive_rag.utils import (
    load_model_and_tokenizer,
    initialize_chromadb_client,
    load_embedding_model,
    get_device,
    load_config
)
from curriculum_compass.naive_rag.course_retriever import CourseRAGPipeline
from curriculum_compass.naive_rag.review_retriever import ReviewsRAGPipeline
from curriculum_compass.naive_rag.reranker import Reranker
from curriculum_compass.naive_rag.retriever_utils import load_course_data
from curriculum_compass.agentic_rag.orchestrator import AgentOrchestrator

# Initialize weave
weave.init(project_name="Agentic_RAG_System")

def main():
    # Load configurations
    config = load_config()
    
    # Initialize components
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize ChromaDB
    chromadb_client = initialize_chromadb_client("../naive_rag/chromadb")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['llm'])
    
    # Initialize Course RAG (reusing your existing setup)
    reranker = Reranker(config['reranker_model_name'], device)
    course_data = load_course_data(config['course_data_path'])
    course_rag = CourseRAGPipeline(reranker)
    course_rag.intiliaze_course_search_system(course_data)
    
    # Initialize Review RAG (reusing your existing setup)
    embedding_model = load_embedding_model(config['embedding_model_name'])
    collection = chromadb_client.get_or_create_collection("naive_rag_embeddings")
    review_rag = ReviewsRAGPipeline(embedding_model, collection, reranker)
    
    # Initialize the Orchestrator
    orchestrator = AgentOrchestrator(
        model=model,
        tokenizer=tokenizer,
        course_rag=course_rag,
        review_rag=review_rag
    )

    # Test the system
    async def test_query():
        query = "How difficult is Algorithms course?"
        result = await orchestrator.process_query(query)
        print("\nQuery:", query)
        print("\nProcessing History:")
        for stage in result.get("processing_history", []):
            print(f"- {stage}")
        print("\nFinal Response:", result.get("final_response"))

    # Run with weave tracing
    with weave.attributes({'user_id': 'test_user', 'env': 'testing'}):
        import asyncio
        asyncio.run(test_query())

if __name__ == "__main__":
    main()