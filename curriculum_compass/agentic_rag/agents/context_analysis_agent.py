import json
from typing import Dict, Any
from .base_agent import BaseAgent

class ContextAnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing if retrieved information sufficiently answers the query
    and providing feedback for improving retrieval if needed.

    Enhancements:
    1. Includes an 'unanswerable' flag in the LLM output.
    2. Differentiates between partial matches vs. no matches.
    3. Prevents infinite loops by forcing 'is_sufficient' if max_iterations are reached.
    """

    def __init__(self, model, tokenizer, max_iterations: int = 2):
        system_prompt = """You are a context analysis specialist for a course information system.
        Your role is to:
        1. Analyze if the retrieved information answers the original query
        2. Identify any missing crucial information
        3. Suggest improvements for retrieval if needed
        4. Determine if the question is unanswerable or out-of-domain
        5. Ensure comprehensive coverage of query intent
        6. Prevent infinite retrieval loops

        Criteria for 'unanswerable':
        - If the course or professor is clearly non-existent in our domain,
          or repeated attempts yield no relevant info, set "unanswerable": true.
        - Example: Mismatched or invalid CRN, or a professor name not in the database, etc.

        Provide a JSON response with the fields described below. If you see partial data or mismatch
        (like professor doesn't match the course), mention it in 'missing_aspects' or the
        'analysis_reasoning'. If you believe a second retrieval pass might help, set 'is_sufficient' to false
        and provide 'feedback' with suggestions.

        Make sure to only respond with valid JSON and no additional commentary.
        """

        super().__init__(model, tokenizer, system_prompt)
        self.max_iterations = max_iterations

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if retrieved information sufficiently answers the query.

        Args:
            input_data: Dictionary containing retrieved info and previous processing data

        Returns:
            Dictionary containing original input plus:
            - context_analysis: LLM-based analysis of completeness
            - needs_improvement: Boolean if retrieval needs more attempts
            - unanswerable: Boolean if the LLM deems no further retrieval can succeed
            - improvement_feedback: Suggestions for refinement (if any)
        """
        # If there's no retrieved info at all, we can still pass it along 
        # but presumably that's a strong sign of potential unanswerability or second retrieval attempt.
        if not input_data.get("retrieved_info"):
            return input_data

        # Check iteration count to prevent infinite loops
        current_iteration = input_data.get("retrieval_iteration", 0)
        if current_iteration >= self.max_iterations:
            # Force 'is_sufficient' after max iterations to avoid infinite loop
            return {
                **input_data,
                "context_analysis": {
                    "is_sufficient": True,
                    "missing_aspects": [],
                    "has_partial_info": True,
                    "unanswerable": False,
                    "message": "Max retrieval iterations reached - forcing exit."
                },
                "needs_improvement": False,
                "unanswerable": False,
                "stage": "context_analysis"
            }

        #############
        # BUILD PROMPT
        #############
        # We instruct the LLM to carefully examine the retrieved data, see if it's partial, 
        # fully sufficient, or nonexistent, and decide if it's fixable or unanswerable.

        retrieved_course_info = input_data["retrieved_info"].get("course_info", [])
        retrieved_review_info = input_data["retrieved_info"].get("review_info", [])

        analysis_prompt = f"""
        Original Query: {input_data.get('original_query', '')}
        Intent Analysis: {json.dumps(input_data.get('intent_analysis', {}), indent=2)}

        Retrieved Course Info: {json.dumps(retrieved_course_info, indent=2)}
        Retrieved Review Info: {json.dumps(retrieved_review_info, indent=2)}

        Please analyze the retrieval results. Consider:
        - Whether we have enough data to fully address the query
        - Whether there's partial data (e.g., course exists but professor doesn't match)
        - Whether it appears no relevant data exists for this query
        - Whether repeated attempts might help (spelling correction, fuzzy match, etc.)
        - If you conclude it's out-of-domain or definitely not in the database, mark "unanswerable": true

        Return strictly valid JSON with the structure:
        {{
            "is_sufficient": boolean,
            "missing_aspects": [
                // list any crucial info that is missing 
                // e.g. "professor mismatch", "no schedule found", etc.
            ],
            "has_partial_info": boolean, 
            "unanswerable": boolean,
            "feedback": {{
                "query_suggestions": {{
                    "course_db": "improved query text or empty if not needed",
                    "review_db": "improved query text or empty if not needed"
                }},
                "retrieval_suggestions": {{
                    "course_k": number or null,
                    "review_k": number or null,
                    "reasoning": "why these adjustments might help or empty"
                }}
            }},
            "analysis_reasoning": "Short text explaining your conclusion"
        }}
        """

        ################
        # CALL THE LLM
        ################
        try:
            response_text = await self.generate_llm_response(analysis_prompt)
            analysis_result = json.loads(response_text)

            # Decide if we need more retrieval attempts
            # We only do further attempts if is_sufficient=False, unanswerable=False
            needs_improvement = (
                not analysis_result.get("is_sufficient", False)
                and not analysis_result.get("unanswerable", False)
            )

            return {
                **input_data,
                "context_analysis": analysis_result,
                "needs_improvement": needs_improvement,
                "unanswerable": analysis_result.get("unanswerable", False),
                "improvement_feedback": analysis_result.get("feedback", {}),
                "stage": "context_analysis"
            }

        except json.JSONDecodeError:
            # If JSON parsing fails, just mark it sufficient to avoid loop
            return {
                **input_data,
                "context_analysis": {
                    "is_sufficient": True,
                    "missing_aspects": ["Error parsing context analysis response"],
                    "has_partial_info": True,
                    "unanswerable": False,
                    "analysis_reasoning": "LLM response was not valid JSON."
                },
                "needs_improvement": False,
                "unanswerable": False,
                "stage": "context_analysis"
            }

        except Exception as e:
            # General exception fallback
            return {
                **input_data,
                "context_analysis": {
                    "is_sufficient": True,
                    "missing_aspects": [f"Context analysis error: {str(e)}"],
                    "has_partial_info": True,
                    "unanswerable": False,
                    "analysis_reasoning": "An exception occurred during analysis."
                },
                "needs_improvement": False,
                "unanswerable": False,
                "stage": "context_analysis"
            }