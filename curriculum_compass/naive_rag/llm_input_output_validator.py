from utils import load_model_and_tokenizer, generate_llm_response
from llm_guard import LLMGuard
import weave

class LLMInputOutputValidator:
    def __init__(self, model_name: str, device: str):
        """
        Initialize the QueryValidator.
        Note: We don't manually move the model to device since it's handled by Accelerate
        """
        self.model_name = model_name
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.device = device
        self.model.eval()
        self.guard = LLMGuard(model_name)

    @weave.op(name="validate_llm_input")
    def validate_input(self, user_query: str) -> str:
        """
        Classify whether a user query is relevant or not relevant
        to Northeastern courses/professors. Returns 'RELEVANT' or 'NOT RELEVANT'.
        """

        is_valid, reason, results = self.guard.validate_input(user_query)
        if not is_valid:
            return False, reason, results



        system_prompt = """
You are a helpful AI system tasked with filtering user questions about Northeastern University courses, professors, and course reviews.

### Relevancy Rules
- Relevant questions are those about:
  • Course offerings, schedules, prerequisites, or location (campus vs. online).
  • Professor/faculty information (e.g., who is teaching, professor's teaching style).
  • Opinions or reviews about the course or professor (e.g., workload, grading difficulty).
  • Past or present course reviews (e.g., "Has this course been offered in the past? How were the reviews?").
  • Anything else directly related to Northeastern courses or professors.

- Irrelevant questions:
  • Topics unrelated to Northeastern courses or professors (e.g., weather, jokes, cooking).
  • Personal advice not connected to Northeastern's courses/professors.
  • Any query that does not pertain to course data or professor data at Northeastern.

Your output:
- Respond EXACTLY with 'RELEVANT' if the question is about Northeastern courses, professors, or reviews (including workload, grading, difficulty).
- Respond EXACTLY with 'NOT RELEVANT' if it is off-topic.

### Examples
1) User query: "Which professor is teaching CS1800 next semester?"
   Answer: RELEVANT
2) User query: "How do I bake a chocolate cake?"
   Answer: NOT RELEVANT
3) User query: "How much workload does CS1800 typically have?"
   Answer: RELEVANT
4) User query: "How is Professor Karl Lieberherr in terms of grading?"
   Answer: RELEVANT
5) User query: "What is the capital of France?"
   Answer: NOT RELEVANT
6) User query: "Has Data Structures been offered previously? Any reviews about difficulty?"
   Answer: RELEVANT
"""
        # Use generate_llm_response but don't modify model device placement
        response = generate_llm_response(
            system_prompt=system_prompt,
            query=user_query,
            retrieved_docs=[],  
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        is_relevant = response.upper() == "RELEVANT"
        if not is_relevant:
            return False, "Query is not relevant to Northeastern courses or professors", results

        return True, "Query passes all validation checks", results
    

    @weave.op(name="validate_llm_output")
    def validate_output(self, output_text: str, input_text: str):
        """
        Validate generated output using LLM Guard.
        """
        return self.guard.validate_output(output_text, input_text)

