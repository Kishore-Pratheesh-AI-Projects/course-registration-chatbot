from typing import Dict, Any
from .agents.validation_agent import ValidationAgent
from .agents.intent_agent import IntentAgent
from .agents.query_enhancement_agent import QueryEnhancementAgent
from .agents.dynamic_retrieval_agent import DynamicRetrievalAgent
from .agents.context_analysis_agent import ContextAnalysisAgent
from .agents.response_agent import ResponseAgent

class AgentOrchestrator:
    """
    Orchestrates the flow between different agents in the multi-agent system.
    Manages the entire pipeline from query validation to final response generation.
    """
    def __init__(self, model, tokenizer, course_rag, review_rag):
        # Initialize all agents
        self.validation_agent = ValidationAgent(model, tokenizer)
        self.intent_agent = IntentAgent(model, tokenizer)
        self.query_enhancement_agent = QueryEnhancementAgent(model, tokenizer)
        self.dynamic_retrieval_agent = DynamicRetrievalAgent(
            model, tokenizer, course_rag, review_rag
        )
        self.context_analysis_agent = ContextAnalysisAgent(model, tokenizer)
        self.response_agent = ResponseAgent(model, tokenizer)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the entire agent pipeline.

        Args:
            query: User's query string

        Returns:
            Dictionary containing final response and processing details
        """
        # Initialize processing state
        state = {"query": query, "processing_history": []}

        try:
            # 1. Validation
            state = await self.validation_agent.process(state)
            if not state.get("valid", False):
                return self._create_error_response(
                    state, "Invalid query: " + state.get("reason", "")
                )

            # 2. Intent Analysis
            state = await self.intent_agent.process(state)
            if state.get("error"):
                return self._create_error_response(
                    state, "Error analyzing query intent"
                )

            # 3. Query Enhancement
            state = await self.query_enhancement_agent.process(state)
            if state.get("error"):
                return self._create_error_response(
                    state, "Error enhancing query"
                )

            # 4. Initial Retrieval Loop
            max_retrieval_attempts = 4
            retrieval_attempt = 0

            while retrieval_attempt < max_retrieval_attempts:
                # 1. Dynamic Retrieval
                state = await self.dynamic_retrieval_agent.process(state)
                # 2. Context Analysis
                state = await self.context_analysis_agent.process(state)

                # If the context analysis sets unanswerable=True, break immediately
                if state.get("unanswerable"):
                    break

                # If no improvement needed or we are at the last attempt, break
                if not state.get("needs_improvement", False) or retrieval_attempt >= max_retrieval_attempts - 1:
                    break

                # Otherwise, refine queries
                feedback = state["improvement_feedback"]
                state = await self.query_enhancement_agent.process_feedback(state, feedback)
                retrieval_attempt += 1

            # 5. Final Response Generation
            state = await self.response_agent.process(state)
            if state.get("error"):
                return self._create_error_response(
                    state, "Error generating response"
                )

            return state

        except Exception as e:
            return self._create_error_response(
                state, f"System error: {str(e)}"
            )

    def _create_error_response(self, state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            **state,
            "error": error_message,
            "final_response": f"I apologize, but I encountered an error: {error_message}. Please try rephrasing your question.",
            "processing_history": state.get("processing_history", []) + ["Error: " + error_message]
        }

    def _log_state(self, state: Dict[str, Any], stage: str):
        """Log the current state for debugging."""
        state["processing_history"] = state.get("processing_history", []) + [
            f"{stage}: {state.get('stage', 'unknown')}"
        ]
        return state