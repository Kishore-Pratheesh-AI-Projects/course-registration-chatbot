import weave
from course_rag import CourseRAGPipeline
from naive_reviews_rag import NaiveReviewsRAGPipeline
from re_ranker import Reranker

# Initialize weave
weave.init(project_name="Integrated_RAG_System")

class IntegratedRAGPipeline:

    # TODO: 1. Combine the system instructions from both CourseRAGPipeline and NaiveReviewsRAGPipeline and then write a 
    #           unified SYSTEM_INSTRUCTION for the IntegratedRAGPipeline
    #       2. Have the Course_RAG_System and Naive_RAG_Reviews return the re-ranked documents and get rid of the 
    #           LLM response generation from each of them. Instead, have the IntegratedRAGPipeline rerank the combined them and 
    #           generate the response using the LLM from CourseRAGPipeline.

    
    SYSTEM_INSTRUCTION = """
    """

    def __init__(self, course_rag: CourseRAGPipeline, review_rag: NaiveReviewsRAGPipeline):
        self.course_rag = course_rag
        self.review_rag = review_rag
        self.final_reranker = Reranker() 
        
    @weave.op(name="get_course_info")
    def get_course_information(self, query: str, top_k: int = 5):
        """Get relevant course information"""
        return self.course_rag.retrieve_courses(query, top_k=top_k)
        
    @weave.op(name="get_reviews")
    def get_reviews(self, query: str, top_k: int = 5):
        """Get relevant reviews"""
        return self.review_rag.retrieve(query, top_k=top_k)[0]  # Assuming similar structure to course_rag
        
    @weave.op(name="combine_and_rerank")
    def combine_and_rerank(self, query: str, course_docs: list, review_docs: list, final_k: int = 5):
        """Combine and rerank all documents"""
        # Combine both types of documents with clear separation
        combined_docs = []
        for doc in course_docs:
            combined_docs.append(f"[COURSE INFO] {doc}")
        for doc in review_docs:
            combined_docs.append(f"[STUDENT REVIEW] {doc}")
            
        # Rerank the combined documents
        try:
            reranked_docs = self.final_reranker.rerank(query, combined_docs, top_k=final_k)
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
        return self.course_rag.generate_response(query, combined_docs)
        
    @weave.op(name="process_integrated_query")
    def __call__(self, query: str, course_k: int = 5, review_k: int = 5, final_k: int = 5):
        """Process a query through the integrated pipeline"""
        print(f"Processing query: {query}")
        
        # Get course information and reviews in parallel
        print("Retrieving course information and reviews...")
        course_docs = self.get_course_information(query, top_k=course_k)
        review_docs = self.get_reviews(query, top_k=review_k)
        
        # Combine and rerank
        print("Combining and reranking all documents...")
        combined_docs = self.combine_and_rerank(query, course_docs, review_docs, final_k=final_k)
        
        # Generate final response
        print("Generating integrated response...")
        response = self.generate_response(query, combined_docs)
        
        return response

def main():
    # Initialize individual RAG pipelines
    course_rag = CourseRAGPipeline()
    #TODO : Initialize the NaiveReviewsRAGPipeline with the appropriate parameters
    review_rag = NaiveReviewsRAGPipeline(embedding_model, collection, model, tokenizer)
    
    # Initialize integrated pipeline
    integrated_rag = IntegratedRAGPipeline(course_rag, review_rag)
    
    # Example usage with weave tracing
    with weave.attributes({'user_id': 'test_user', 'env': 'testing'}):
        query = "What do students say about the workload in AI courses, and which ones are currently available?"
        response = integrated_rag(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()