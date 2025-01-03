import json
from typing import Dict, Any
from .base_agent import BaseAgent

class IntentAgent(BaseAgent):
    """
    Agent responsible for understanding query intent and mapping it to available data structure.
    """
    def __init__(self, model, tokenizer):
        system_prompt = """You are an intent analyzer for NEU course queries. 
        The system has access to two specific types of information:

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
        Question: [question about course/instructor]
        Review: [student feedback]

        Your job is to analyze queries and determine exactly what information is needed from these structured data sources."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not input_data.get("valid", False):
            return input_data

        query = input_data.get("original_query", "")
        
        # intent_prompt = f"""
        # Query: {query}

        # Analyze this query and return a JSON response with:
        # {{
        #     "primary_intent": "course_info|student_experience|combined",
        #     "required_fields": {{
        #         "course_db": [
        #             // List specific sections needed from course format
        #             // Options: "course metadata", "location", "schedule", "instructor", "course details", "description"
        #         ],
        #         "review_db": [
        #             // List if student reviews are needed
        #             // Options: "general_feedback", "difficulty_feedback", "professor_feedback"
        #         ]
        #     }},
        #     "priority": {{
        #         "course_info_weight": number (0-10),
        #         "review_info_weight": number (0-10)
        #     }},
        #     "specific_aspects": [
        #         // List specific information needed
        #         // e.g., "prerequisites", "schedule", "difficulty", "teaching_style"
        #     ],
        #     "reasoning": "explain why this information is needed for the query"
        # }}
        # """


        intent_prompt = f"""
Query: {query}

Analyze this query and return a JSON response with:
{{
    "primary_intent": "course_info|student_experience|combined",
    "required_fields": {{
        "course_db": [
            // List all specific sections needed from the course format based on the query
            // Must match exactly with course database sections
        ],
        "review_db": [
            // List specific types of review information needed
            // Be specific about what kind of feedback or experiences are relevant
        ]
    }},
    "priority": {{
        "course_info_weight": number (0-10),
        "review_info_weight": number (0-10)
    }},
    "specific_aspects": [
        // List ALL specific information points needed to answer this query
        // Be detailed and exact about what information points are required
    ],
    "reasoning": "explain why this information is needed for the query"
}}
"""

        try:
            response = await self.generate_llm_response(intent_prompt)
            intent_analysis = json.loads(response)
            
            return {
                **input_data,
                "intent_analysis": intent_analysis,
                "stage": "intent_analysis",
                "processing_status": {
                    "needs_course_info": len(intent_analysis["required_fields"]["course_db"]) > 0,
                    "needs_reviews": len(intent_analysis["required_fields"]["review_db"]) > 0,
                    "priority_source": "course_db" if intent_analysis["priority"]["course_info_weight"] > 
                                     intent_analysis["priority"]["review_info_weight"] else "review_db"
                }
            }

        except json.JSONDecodeError:
            return {
                **input_data,
                "intent_analysis": None,
                "error": "Failed to parse intent analysis",
                "stage": "intent_analysis"
            }
        
        except Exception as e:
            return {
                **input_data,
                "intent_analysis": None,
                "error": f"Intent analysis error: {str(e)}",
                "stage": "intent_analysis"
            }