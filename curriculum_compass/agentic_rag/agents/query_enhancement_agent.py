import json
from typing import Dict, Any
from .base_agent import BaseAgent

class QueryEnhancementAgent(BaseAgent):
    """
    Agent responsible for enhancing queries to better match database structures
    and improve retrieval quality based on intent.
    """
    def __init__(self, model, tokenizer):
        # system_prompt = """You are a query enhancement specialist for a course registration system.
        
        # 1. Course Database Format:
        # === course metadata === 
        # - course code
        # - crn
        # - title
        # === location === 
        # - campus
        # - format (in-person/online)
        # === schedule === 
        # - days
        # - time
        # === instructor === 
        # - professor name
        # === course details ===
        # - term
        # - prerequisites
        # === description ===
        # - course description

        # 2. Review Database Format:
        # Metadata:
        # - CRN
        # - Course Name
        # - Instructor
        # - Course Number
        # Question: [question about course/instructor/] 
        # Review: [student feedback]

        # Your job is to enhance queries to match these formats and improve retrieval."""

        system_prompt = """You are a query enhancement specialist for a course registration system.
        
        1. Course Database Format:
        === course metadata === 
        - course code
        - crn
        - title
        === location === 
        - campus
        - format (in-person/online)
        === schedule === 
        - days
        - time
        === instructor === 
        - professor name
        === course details ===
        - term
        - prerequisites
        === description ===
        - course description

        2. Review Database Format:
        Metadata:
        - CRN
        - Course Name
        - Instructor
        - Course Number
        Question: Reviews can contain feedback about:
            - Course difficulty and workload
            - Professor teaching style and effectiveness
            - Assignment and exam experiences
            - Course content and materials
            - Class structure and organization
            - Student preparation advice
            - Prerequisites helpfulness
            - Career relevance
            - Overall course experience
            - Recommendations for future students
        Review: [detailed student feedback on any of these aspects]

        Your job is to enhance queries to match these formats and improve retrieval."""

        super().__init__(model, tokenizer, system_prompt)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance queries based on intent analysis and database structures.

        Args:
            input_data: Dictionary containing intent analysis and previous info

        Returns:
            Dictionary containing original input plus:
            - enhanced_queries: Optimized queries for each database
        """
        if not input_data.get("intent_analysis"):
            return input_data

        query = input_data.get("original_query", "")
        intent = input_data.get("intent_analysis", {})
        
        enhancement_prompt = f"""
        Original Query: {query}
        Intent Analysis: {json.dumps(intent, indent=2)}

        Create optimized search queries for both databases. Return JSON:
        {{
            "course_db_query": "enhanced query for course database",
            "review_db_query": "enhanced query for review database",
            "enhancement_reasoning": "explanation of enhancements made",
            "focus_aspects": ["specific aspects to look for in retrieved content"]
        }}
        """

        try:
            response = await self.generate_llm_response(enhancement_prompt)
            enhanced_queries = json.loads(response)
            
            return {
                **input_data,
                "enhanced_queries": enhanced_queries,
                "stage": "query_enhancement"
            }

        except json.JSONDecodeError:
            return {
                **input_data,
                "enhanced_queries": None,
                "error": "Failed to generate enhanced queries",
                "stage": "query_enhancement"
            }
        
        except Exception as e:
            return {
                **input_data,
                "enhanced_queries": None,
                "error": f"Query enhancement error: {str(e)}",
                "stage": "query_enhancement"
            }

    async def process_feedback(self, input_data: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle feedback from Context Analysis Agent to improve queries.

        Args:
            input_data: Original processing result
            feedback: Feedback from Context Analysis Agent

        Returns:
            Updated processing result with new enhanced queries
        """
        feedback_prompt = f"""
        Original Query: {input_data.get('original_query', '')}
        Previous Enhancement: {json.dumps(input_data.get('enhanced_queries', {}), indent=2)}
        Feedback: {json.dumps(feedback, indent=2)}

        Create improved search queries addressing the feedback. Return JSON:
        {{
            "course_db_query": "improved query for course database",
            "review_db_query": "improved query for review database",
            "enhancement_reasoning": "explanation of new improvements",
            "focus_aspects": ["updated aspects to look for"]
        }}
        """

        try:
            response = await self.generate_llm_response(feedback_prompt)
            improved_queries = json.loads(response)
            
            return {
                **input_data,
                "enhanced_queries": improved_queries,
                "enhancement_iteration": input_data.get("enhancement_iteration", 0) + 1,
                "stage": "query_enhancement_retry"
            }

        except Exception as e:
            return {
                **input_data,
                "error": f"Query enhancement feedback error: {str(e)}",
                "stage": "query_enhancement_retry"
            }