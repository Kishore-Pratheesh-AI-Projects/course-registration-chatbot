import json
from typing import Dict, Any
from .base_agent import BaseAgent

class ResponseAgent(BaseAgent):
    """
    Final agent responsible for generating comprehensive responses based on 
    analyzed context and query intent.
    """
    def __init__(self, model, tokenizer):
        system_prompt = """You are a course advisor for Northeastern University students.
        Your role is to generate comprehensive, clear, and helpful responses using:
        1. The original query intent
        2. Retrieved course information
        3. Student review insights
        
        Focus on providing accurate, well-structured responses that directly address 
        the student's needs. If any information is limited or missing, acknowledge 
        this transparently while providing the best possible guidance with 
        available information."""

        super().__init__(model, tokenizer, system_prompt)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final response based on all collected and analyzed information.

        Args:
            input_data: Dictionary containing all previous processing results

        Returns:
            Dictionary containing original input plus final response
        """
        # Early return if critical data is missing
        if not input_data.get("retrieved_info") or not input_data.get("context_analysis"):
            return {
                **input_data,
                "final_response": "I apologize, but I'm unable to provide a response due to missing information.",
                "stage": "response_generation"
            }

        # Prepare context for response generation
        response_prompt = f"""
        Original Query: {input_data.get('original_query', '')}
        Intent Analysis: {json.dumps(input_data.get('intent_analysis', {}), indent=2)}
        Retrieved Information:
        Course Info: {json.dumps(input_data['retrieved_info']['course_info'], indent=2)}
        Review Info: {json.dumps(input_data['retrieved_info']['review_info'], indent=2)}
        Context Analysis: {json.dumps(input_data.get('context_analysis', {}), indent=2)}

        Generate a comprehensive response that:
        1. Directly addresses the student's question
        2. Incorporates both course information and student experiences
        3. Acknowledges any limitations in available information
        4. Provides additional relevant suggestions or considerations
        5. Maintains a helpful and professional tone

        Return JSON:
        {{
            "response": "the complete response text",
            "has_limitations": boolean,
            "limitation_notes": ["list of any information limitations"],
            "additional_suggestions": ["list of additional helpful suggestions"]
        }}
        """

        try:
            response = await self.generate_llm_response(response_prompt)
            response_data = json.loads(response)
            
            # Format final response
            final_response = response_data["response"]
            
            # Add limitation notes if any
            if response_data.get("has_limitations", False):
                limitations = "\n\nNote: " + ", ".join(response_data.get("limitation_notes", []))
                final_response += limitations
            
            # Add additional suggestions if any
            suggestions = response_data.get("additional_suggestions", [])
            if suggestions:
                suggestions_text = "\n\nAdditional suggestions:\n- " + "\n- ".join(suggestions)
                final_response += suggestions_text

            return {
                **input_data,
                "response_data": response_data,
                "final_response": final_response,
                "stage": "response_generation"
            }

        except json.JSONDecodeError:
            return {
                **input_data,
                "final_response": "I apologize, but I encountered an error while generating the response. Please try rephrasing your question.",
                "error": "Failed to parse response generation",
                "stage": "response_generation"
            }
        
        except Exception as e:
            return {
                **input_data,
                "final_response": "I apologize, but I encountered an error while generating the response. Please try again.",
                "error": f"Response generation error: {str(e)}",
                "stage": "response_generation"
            }