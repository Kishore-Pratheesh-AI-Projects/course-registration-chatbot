
import weave


from curriculum_compass.naive_rag.search_system import CourseSearchSystem


# # Initialize weave
# weave.init(project_name="Course_RAG_System")

class CourseRAGPipeline:
    SYSTEM_INSTRUCTION = """
    You are Curriculum compass, a chatbot which helps Northeastern University students find course offerings of their choice for the Spring 2025 semester.

    You have access to all the course offerings for the Spring 2025 semester, your objective is to use this context to answer student questions.

    Instructions:
    1. Students may not mention the names of the courses properly. Their input could have typo's, mistakes. For example, students could input 'PDP' instead of 
    'Programming Design Paradigm' or they could mention 'Raj Venkat' instead of the full name of the professor 'Rajagopal Venkatesaramani'
    """

    def __init__(self,re_ranker):
        self.course_search_system = CourseSearchSystem()
        # model_name = "Qwen/Qwen2.5-3B-Instruct"
        # self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.re_ranker = re_ranker

    @weave.op(name="retrieve_courses")
    def retrieve(self, query: str, top_k: int = 10):
        """Retrieve relevant course information"""
        results = self.course_search_system.query_courses(query, top_k)
        # Flatten the nested list structure
        return [doc for sublist in results["documents"] for doc in sublist]  

    # @weave.op(name="generate_llm_response")
    # def generate_response(self, query: str, retrieved_docs: list):
    #     """Generate response using the language model"""
    #     context = "\n\n".join(retrieved_docs)  
    #     prompt = f"""Context:{context}, Query: {query}"""

    #     messages = [
    #         {"role": "system", "content": self.SYSTEM_INSTRUCTION},
    #         {"role": "user", "content": prompt}
    #     ]
        
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
        
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    #     generated_ids = self.model.generate(
    #         **model_inputs,
    #         max_new_tokens=4500,
    #         temperature=0.1
    #     )
        
    #     generated_ids = [
    #         output_ids[len(input_ids):] 
    #         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
        
    #     return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    @weave.op(name="rerank_courses")
    def rerank(self, query: str, retrieved_docs: list, top_k: int = 5):
        """Cross-encoder reranking of retrieved documents"""
        try:
            if not retrieved_docs:
                print("Warning: No documents to rerank")
                return []
                
            if len(retrieved_docs) < top_k:
                print(f"Warning: Requested top_k={top_k} but only {len(retrieved_docs)} documents available")
                top_k = len(retrieved_docs)
                
            reranked_docs = self.re_ranker.rerank(query, retrieved_docs, top_k=top_k)
            print(f"Successfully reranked {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            print(f"Error during reranking: {str(e)}")
            # Fall back to original documents if reranking fails
            return retrieved_docs[:top_k]
        
    @weave.op(name="process_course_query")
    def __call__(self, query: str, initial_k: int = 10, final_k: int = 5):
        """Process a query through the RAG pipeline with nested weave tracing"""
        print(f"Processing query: {query}")
        
        print("Retrieving relevant course information...")
        retrieved_docs = self.retrieve(query, top_k=initial_k)

        print("Cross-encoder reranking...")
        reranked_docs = self.rerank(query, retrieved_docs, top_k=final_k)
        
        # print("Generating response...")
        # response = self.generate_response(query, reranked_docs)
        
        return reranked_docs
    
    def intiliaze_course_search_system(self, course_data):
        self.course_search_system.add_course_sentences_to_db(course_data)

# def main():
#     # # Initialize processor and process course data
#     # # course_data = CourseDataProcessor.process_course_data('/Users/pratheeshjp/Documents/course-registration-chatbot/curriculum_compass/data_pipeline/notebooks/data/courses.csv')
    
#     # # Initialize RAG pipeline
#     # rag_pipeline = CourseRAGPipeline()
#     # rag_pipeline.course_search_system.add_course_sentences_to_db(course_data)
    
#     # Example usage with weave tracing
#     with weave.attributes({'user_id': 'test_user', 'env': 'testing'}):
#         query = "Can you suggest courses which are not too hectic and easy to get good grades?"
#         response = rag_pipeline(query)
#         print(f"\nQuery: {query}")
#         print(f"Response: {response}")

# if __name__ == "__main__":
#     main()