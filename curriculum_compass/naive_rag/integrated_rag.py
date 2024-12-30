import weave
from curriculum_compass.naive_rag.llm_input_output_validator import LLMInputOutputValidator
from curriculum_compass.naive_rag.course_retriever import CourseRAGPipeline
from curriculum_compass.naive_rag.review_retriever import ReviewsRAGPipeline
from curriculum_compass.naive_rag.reranker import Reranker
from curriculum_compass.naive_rag.retriever_utils import load_course_data
from curriculum_compass.naive_rag.utils import get_device
from curriculum_compass.naive_rag.utils import load_config
from curriculum_compass.naive_rag.utils import load_embedding_model
from curriculum_compass.naive_rag.utils import generate_llm_response
from curriculum_compass.naive_rag.utils import load_model_and_tokenizer
from curriculum_compass.naive_rag.utils import initialize_chromadb_client



# Initialize weave
weave.init(project_name="Course_RAG_System")

class IntegratedRAGPipeline:
    def __init__(self, course_rag: CourseRAGPipeline, review_rag: ReviewsRAGPipeline,config:dict,device:str): 
        self.course_rag = course_rag
        self.review_rag = review_rag
        self.LLM = config['llm']
        self.model, self.tokenizer = load_model_and_tokenizer(self.LLM)
        self.final_reranker = Reranker(config['reranker_model_name'],device) 
        self.system_prompt = config["system_prompt"]
        
    @weave.op(name="get_course_info")
    def get_course_information(self, query: str, top_k: int = 5):
        """Get relevant course information"""
        return self.course_rag.retrieve_courses(query, top_k=top_k)
        
    @weave.op(name="get_reviews")
    def get_reviews(self, query: str, top_k: int = 5):
        """Get relevant reviews"""
        return self.review_rag.retrieve(query, top_k=top_k)[0]  # Assuming similar structure to course_rag
        
    @weave.op(name="combine_and_rerank_integrated")
    def combine_and_rerank(self, query: str, course_docs: list, review_docs: list, final_k:int):
        """Combine and rerank all documents"""
        # Combine both types of documents with clear separation
        combined_docs = []
        for doc in course_docs:
            combined_docs.append(f"[COURSE INFO] {doc}")
        for doc in review_docs:
            combined_docs.append(f"[STUDENT REVIEW] {doc}")
            
        # Rerank the combined documents
        try:
            reranked_docs = self.final_reranker.rerank(query, combined_docs, final_k)
            print(f"Successfully reranked {len(reranked_docs)} combined documents")
            return reranked_docs
        except Exception as e:
            print(f"Error during final reranking: {str(e)}")
            # Fall back to original combined docs
            return combined_docs[:final_k]
            
    @weave.op(name="generate_integrated_response")
    def generate_response(self, query: str, combined_docs: list):
        """Generate final response using combined context"""
        # Use the course_rag's LLM for response generation
        return generate_llm_response(self.system_prompt, query,combined_docs,self.model,self.tokenizer)
        
    @weave.op(name="process_integrated_query")
    def __call__(self, query: str, course_k, review_k, final_k):
        """Process a query through the integrated pipeline"""
        print(f"Processing query: {query}")
        
        # Get course information and reviews in parallel
        print("Retrieving course information and reviews...")
        # course_docs = self.get_course_information(query, top_k=course_k)
        # review_docs = self.get_reviews(query, top_k=review_k)

        course_docs = self.course_rag(query,course_k,final_k)

        review_docs = self.review_rag(query,review_k,final_k)
        
        # Combine and rerank
        print("Combining and reranking all documents...")
        combined_docs = self.combine_and_rerank(query, course_docs, review_docs, final_k)
        
        # Generate final response
        print("Generating integrated response...")
        response = self.generate_response(query, combined_docs)
        
        return response
    

def main():


# ======= Load Json Configurations =======
    config = load_config()

    chromadb_client = initialize_chromadb_client("./chromadb")


# ===== Initialize the re-ranker ===========
    device = get_device()
    print(f"Using device: {device}")
    reranker = Reranker(config['reranker_model_name'],device)

# ===== Initialize the CourseRagPipeline ===========

    course_data = load_course_data(config['course_data_path'])
    course_rag = CourseRAGPipeline(reranker)
    course_rag.intiliaze_course_search_system(course_data)

# ===== Initialize the NaiveReviewsRAGPipeline ===========

    #TODO : Initialize the NaiveReviewsRAGPipeline with the appropriate parameters
    embedding_model = load_embedding_model(config['embedding_model_name'])
    collection = chromadb_client.get_or_create_collection("naive_rag_embeddings")
    review_rag = ReviewsRAGPipeline(embedding_model, collection,reranker)
    
# ===== Initialize the IntegratedRAGPipeline ===========
    integrated_rag = IntegratedRAGPipeline(course_rag, review_rag,config,device)

# ===== Initlialize the Query Validator ===========
    query_validator = LLMInputOutputValidator(config['query_model_name'],device)

# ===== Example usage of the IntegratedRAGPipeline ===========
    
    # Example usage with weave tracing
    with weave.attributes({'user_id': 'test_user', 'env': 'testing'}):
        query = "How is the weather today?"
        input_status, _, _ = query_validator.validate_input(query)
        if input_status:
            try:
                response = integrated_rag(query,config['course_k'],
                                        config['review_k'],
                                        config['final_k'])
                # print(f"\nQuery: {query}")
                # print(f"Response: {response}")
                output_status, _, _ = query_validator.validate_output(response,query)
                if output_status:
                    print(f"Response: {response}")
                else:
                    #TODO : Add a response for invalid responses
                    # print(f"\{reason}")
                    print("Invalid Response")
            except Exception as e:
                print(f"Error during processing: {str(e)}")
        else:
            #TODO : Add a response for invalid queries
            # print(f"\{reason}")
            print("Invalid Query") 

if __name__ == "__main__":
    main()