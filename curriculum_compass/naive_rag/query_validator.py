from utils import load_model_and_tokenizer, generate_llm_response
import weave

class QueryValidator:
    def __init__(self, model_name: str, device: str):
        """
        Initialize the QueryValidator.
        Note: We don't manually move the model to device since it's handled by Accelerate
        """
        self.model_name = model_name
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.device = device
        # Remove the model.to(device) call since Accelerate handles device placement
        self.model.eval()

    @weave.op(name="validate_input")
    def validate_input(self, user_query: str) -> str:
        """
        Classify whether a user query is relevant or not relevant
        to Northeastern courses/professors. Returns 'RELEVANT' or 'NOT RELEVANT'.
        """
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
        
        classification = response.upper()
        
        if classification not in ["RELEVANT", "NOT RELEVANT"]:
            classification = "NOT RELEVANT"
        return classification

    @weave.op(name="handle_user_query")
    def handle_user_query(self, user_query: str) -> bool:
        """
        Wrapper function that first checks relevancy of the user query.
        If relevant, proceed to retrieval and generation. If not, discard or handle differently.
        """
        classification = self.validate_input(user_query)
        return classification == "RELEVANT"