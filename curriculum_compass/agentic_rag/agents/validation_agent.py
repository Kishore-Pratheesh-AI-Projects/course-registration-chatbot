import json
from typing import Dict, Any
from .base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating if queries are relevant to NEU course registration.
    First line of defense in the system.
    """
    def __init__(self, model, tokenizer):
        system_prompt = """You are a specialized validator for Northeastern University course-related queries.
        Your role is to determine if a query is relevant to:
        1. NEU courses
        2. Course selection
        3. Professor information
        4. Course reviews/experiences
        5. Course registration related questions
        
        You must be strict in validation to prevent irrelevant queries."""

        super().__init__(model, tokenizer, system_prompt)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the input query.

        Args:
            input_data: Dictionary containing "query" key with the user's question

        Returns:
            Dictionary containing:
            - valid: Boolean indicating if query is valid
            - reason: String explanation if invalid
            - original_query: Original query text
        """
        query = input_data.get("query", "")
        
        validation_prompt = f"""
        Query: {query}

        Determine if this query is relevant to NEU courses or course selection.
        Return response in JSON format:
        {{
            "is_valid": boolean,
            "reason": "explanation if invalid, otherwise empty string"
        }}
        """

        try:
            response = await self.generate_llm_response(validation_prompt)
            result = json.loads(response)
            
            return {
                "valid": result["is_valid"],
                "reason": result.get("reason", ""),
                "original_query": query,
                "stage": "validation"
            }

        except json.JSONDecodeError:
            return {
                "valid": False,
                "reason": "Error in validation process",
                "original_query": query,
                "stage": "validation"
            }

        except Exception as e:
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}",
                "original_query": query,
                "stage": "validation"
            }