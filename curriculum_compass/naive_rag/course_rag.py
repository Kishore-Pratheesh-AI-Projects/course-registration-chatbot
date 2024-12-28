
import weave
from utils import load_model_and_tokenizer
from data_processor import CourseDataProcessor
from search_system import CourseSearchSystem

# Initialize weave
weave.init(project_name="Course_RAG_System")

class CourseRAGPipeline:
    SYSTEM_INSTRUCTION = """
    You are Curriculum compass, a chatbot which helps Northeastern University students find course offerings of their choice for the Spring 2025 semester.

    You have access to all the course offerings for the Spring 2025 semester, your objective is to use this context to answer student questions.

    Instructions:
    1. Students may not mention the names of the courses properly. Their input could have typo's, mistakes. For example, students could input 'PDP' instead of 
    'Programming Design Paradigm' or they could mention 'Raj Venkat' instead of the full name of the professor 'Rajagopal Venkatesaramani'
    """

    def __init__(self):
        self.course_search_system = CourseSearchSystem()
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)

    @weave.op(name="retrieve_courses")
    def retrieve_courses(self, query: str, top_k: int = 10):
        """Retrieve relevant course information"""
        results = self.course_search_system.query_courses(query, top_k)
        return results["documents"]

    @weave.op(name="generate_llm_response")
    def generate_response(self, query: str, retrieved_docs: list):
        """Generate response using the language model"""
        context = "\n\n".join([doc for sublist in retrieved_docs for doc in sublist])
        prompt = f"""Context:{context}, Query: {query}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4500,
            temperature=0.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @weave.op(name="process_query")
    def __call__(self, query: str, top_k: int = 10):
        """Process a query through the RAG pipeline with nested weave tracing"""
        print(f"Processing query: {query}")
        
        print("Retrieving relevant course information...")
        retrieved_docs = self.retrieve_courses(query, top_k)
        
        print("Generating response...")
        response = self.generate_response(query, retrieved_docs)
        
        return response

def main():
    # Initialize processor and process course data
    course_data = CourseDataProcessor.process_course_data('../data_pipeline/notebooks/data/courses.csv')
    
    # Initialize RAG pipeline
    rag_pipeline = CourseRAGPipeline()
    rag_pipeline.course_search_system.add_course_sentences_to_db(course_data)
    
    # Example usage with weave tracing
    with weave.attributes({'user_id': 'test_user', 'env': 'testing'}):
        query = "Are there any prerequisite courses for Artificial Intelligence for Human Computer Interaction?"
        response = rag_pipeline(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()