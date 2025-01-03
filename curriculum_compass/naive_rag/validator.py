from utils import load_model_and_tokenizer, generate_llm_response
from guard import LLMGuard
import weave
from typing import List

class Validator:
    def __init__(self, model_name: str, device: str,banned_substrings: List[str],relevance_prompt:str):
        """
        Initialize the QueryValidator.
        Note: We don't manually move the model to device since it's handled by Accelerate
        """
        self.model_name = model_name
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.device = device
        self.model.eval()
        self.guard = LLMGuard(banned_substrings)
        self.relevency_prompt = relevance_prompt


    @weave.op(name="validate_llm_input")
    def validate_input(self, user_query: str) -> str:
        """
        1) Validate user_query with LLMGuard
        2) If it fails, generate an LLM-based user-friendly explanation
        3) If it passes, do relevancy check
        4) Return outcome
        """
        # ---- Step 1: Validate input with LLMGuard
        is_valid, guard_reason, guard_results = self.guard.validate_input(user_query)
        if not is_valid:
            # Generate a user-friendly explanation for why the input was rejected
            user_friendly_response = self.generate_explanation_for_guard_fail(
                guard_reason, guard_results
            )
            return False, user_friendly_response
        
        # ---- Step 2: Validate the relevancy of the query
        is_relevant = self.is_relavent(user_query)
        if not is_relevant:
            # Generate a user-friendly explanation for why the query is not relevant
            user_friendly_response = self.generate_explanation_for_relevance_fail(user_query)
            return False, user_friendly_response

        return True, "Query is valid and relevant"
        

    @weave.op(name="is_relavent")
    def is_relavent(self, user_query: str) -> bool:
        """
        Check if a user query is relevant to Northeastern courses/professors.
        """
        response = generate_llm_response(
            system_prompt=self.relevency_prompt,
            query=user_query,
            retrieved_docs=[],  
            model=self.model,
            tokenizer=self.tokenizer
        )
        return response.upper() == "RELEVANT"
    

    @weave.op(name="generate_explanation_for_guard_fail")
    def generate_explanation_for_guard_fail(self, rejection_reason: str, guard_results: dict) -> str:
        """
        Uses an LLM to produce a structured explanation for query rejection with suggested alternatives.
        """
        system_prompt = (
            "You are an AI designed to explain content policy rejections for queries about NEU courses and professors. "
            "For each flagged query, respond with exactly three lines:\n"
            "1. First line must state: 'CONTENT POLICY VIOLATION'\n"
            "2. Second line: Brief, friendly explanation of why the content was flagged\n"
            "3. Third line: 'Suggested rephrasing: ' followed by a more appropriate way to ask about:\n"
            "   - Course content or structure\n"
            "   - Teaching methods\n"
            "   - Academic experiences\n"
            "   - Professor expertise\n"
            "Focus on academic and professional language."
        )

        user_query = f"""
        Rejection reason: {rejection_reason}
        Scanner details: {guard_results}

        Instructions:
        Generate a 3-line response:
        Line 1: 'CONTENT POLICY VIOLATION'
        Line 2: Explain the issue in a helpful, constructive way
        Line 3: Suggest a better way to ask about the same topic using academic language
        
        Keep the total response under 50 words.
        """

        response = generate_llm_response(
            system_prompt=system_prompt,
            query=user_query,
            retrieved_docs=[],
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.7
        )

        return response.strip()

        


    @weave.op(name="generate_explanation_for_relevance_fail")
    def generate_explanation_for_relevance_fail(self, query) -> str:
        system_message = (
            "You are an AI designed to provide information on Northeastern University courses and professors. "
            "For off-topic queries, respond with exactly three lines:\n"
            "1. First line must be exactly: 'NOT RELEVANT'\n"
            "2. Second line: explanation properly why the query is irrelavent \n"
            "3. Third line: 'Suggested question: ' followed by a question about NEU courses/professors. "
            "Questions should vary between these types:\n"
            "   - Course content questions (e.g., 'What topics are covered in NEU's Machine Learning course?')\n"
            "   - Teaching style questions (e.g., 'How does Professor X teach Database Management?')\n"
            "   - Course reviews/experience (e.g., 'How challenging is the Algorithms course at NEU?')\n"
            "   - Course structure (e.g., 'What projects are included in Software Engineering?')\n"
            "   - Professor expertise (e.g., 'Which professors specialize in AI at NEU?')\n"
            "\nFocus on these CS topics:\n"
            "   - Machine Learning\n"
            "   - Algorithms\n"
            "   - Database Management Systems\n"
            "   - Artificial Intelligence\n"
            "   - Data Structures\n"
            "   - Software Engineering\n"
            "   - Computer Networks\n"
            "   - Operating Systems"
        )

        user_message = f"""
        User query: {query}

        Instructions:
        Generate a 3-line response with:
        Line 1: 'NOT RELEVANT'
        Line 2: Explain why this query isn't about NEU academics
        Line 3: Suggest a question about NEU courses/professors that covers either:
            - Course content and topics
            - Teaching methods and style
            - Student experiences and reviews
            - Course structure and assignments
            - Professor expertise and approach
        Make the suggestion feel natural and focused on what students might want to know.
        """

        response = generate_llm_response(
            system_prompt=system_message,
            query=user_message,
            retrieved_docs=[],
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.9 # Lower the temperature for more controlled output
        )
        return response.strip()
    


    @weave.op(name="validate_llm_output")
    def validate_output(self, output_text: str, input_text: str):
        """
        Validate generated output using LLM Guard.
        """
        return self.guard.validate_output(output_text, input_text)

