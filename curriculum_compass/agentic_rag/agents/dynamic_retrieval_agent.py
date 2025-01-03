import json
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

class DynamicRetrievalAgent(BaseAgent):
    """
    Agent responsible for determining retrieval strategy and executing retrieval
    from both course and review databases.
    """
    def __init__(self, model, tokenizer, course_rag, review_rag):
        system_prompt = """You are a retrieval strategist for a course information system.
        Your role is to determine how much information to retrieve from each database
        based on the query intent and requirements.
        
        Consider:
        1. Primary intent of query
        2. Balance between course info and reviews needed
        3. Specific aspects being asked about
        4. Optimal retrieval amounts (k values) for each database"""
        
        super().__init__(model, tokenizer, system_prompt)
        self.course_rag = course_rag
        self.review_rag = review_rag
        self.TOTAL_K = 15 

    async def determine_retrieval_strategy(self, 
                                        query: str, 
                                        intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal retrieval amounts for each database."""
        
        strategy_prompt = f"""
        Query: {query}
        Intent Analysis: {json.dumps(intent_analysis, indent=2)}

        Determine optimal retrieval strategy. The total number of results (course_k + review_k) 
        must equal exactly {self.TOTAL_K}. Return JSON:
        {{
            "course_k": number (1-{self.TOTAL_K}),
            "review_k": number (1-{self.TOTAL_K}),
            "reasoning": "explanation for this distribution",
            "priority_source": "course_db|review_db|balanced"
        }}
        Note: course_k + review_k must equal {self.TOTAL_K}
        """

        try:
            response = await self.generate_llm_response(strategy_prompt)
            return json.loads(response)
        except Exception as e:
            # Default balanced strategy if determination fails
            return {
                "course_k": 5,
                "review_k": 5,
                "reasoning": "Default balanced strategy due to error",
                "priority_source": "balanced"
            }

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine strategy and retrieve information from both databases.

        Args:
            input_data: Dictionary containing enhanced queries and previous info

        Returns:
            Dictionary containing original input plus:
            - retrieval_strategy: Determined retrieval parameters
            - retrieved_info: Information from both databases
        """
        if not input_data.get("enhanced_queries"):
            return input_data

        try:
            # Get retrieval strategy
            strategy = await self.determine_retrieval_strategy(
                input_data["original_query"],
                input_data["intent_analysis"]
            )

            # Get enhanced queries
            enhanced_queries = input_data["enhanced_queries"]
            
            # Retrieve from both databases
            course_results = await self.course_rag(
                enhanced_queries["course_db_query"],
                k=strategy["course_k"]
            )
            
            review_results = await self.review_rag(
                enhanced_queries["review_db_query"],
                k=strategy["review_k"]
            )

            return {
                **input_data,
                "retrieval_strategy": strategy,
                "retrieved_info": {
                    "course_info": course_results,
                    "review_info": review_results
                },
                "stage": "dynamic_retrieval"
            }

        except Exception as e:
            return {
                **input_data,
                "error": f"Retrieval error: {str(e)}",
                "stage": "dynamic_retrieval"
            }

    async def adjust_retrieval(self, 
                             input_data: Dict[str, Any], 
                             feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust retrieval based on feedback from Context Analysis Agent.
        
        Args:
            input_data: Original processing result
            feedback: Feedback about missing information
        
        Returns:
            Updated processing result with new retrieved information
        """
        try:
            # Adjust strategy based on feedback
            adjustment_prompt = f"""
            Original Strategy: {json.dumps(input_data.get('retrieval_strategy', {}), indent=2)}
            Feedback: {json.dumps(feedback, indent=2)}

            Determine adjusted retrieval strategy. Return JSON:
            {{
                "course_k": number (1-15),
                "review_k": number (1-15),
                "reasoning": "explanation for adjustments"
            }}
            """
            
            response = await self.generate_llm_response(adjustment_prompt)
            adjusted_strategy = json.loads(response)

            # Perform adjusted retrieval
            new_course_results = await self.course_rag(
                input_data["enhanced_queries"]["course_db_query"],
                k=min(adjusted_strategy["course_k"], self.max_k)
            )
            
            new_review_results = await self.review_rag(
                input_data["enhanced_queries"]["review_db_query"],
                k=min(adjusted_strategy["review_k"], self.max_k)
            )

            return {
                **input_data,
                "retrieval_strategy": adjusted_strategy,
                "retrieved_info": {
                    "course_info": new_course_results,
                    "review_info": new_review_results
                },
                "retrieval_iteration": input_data.get("retrieval_iteration", 0) + 1,
                "stage": "dynamic_retrieval_retry"
            }

        except Exception as e:
            return {
                **input_data,
                "error": f"Retrieval adjustment error: {str(e)}",
                "stage": "dynamic_retrieval_retry"
            }