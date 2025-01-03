from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    All other agents will inherit from this base class.
    """
    def __init__(self, model, tokenizer, system_prompt: str):
        """
        Initialize base agent.
        
        Args:
            model: The LLM model
            tokenizer: The model's tokenizer
            system_prompt: Specific instructions for this agent's role
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        Must be implemented by each specific agent.

        Args:
            input_data: Dictionary containing input data for processing
            
        Returns:
            Dictionary containing processed results
        """
        pass

    async def generate_llm_response(self, user_prompt: str) -> str:
        """
        Generate response using the LLM.
        
        Args:
            user_prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.2
        )
        
        decoded = self.tokenizer.batch_decode(outputs[:, len(model_inputs.input_ids[0]):])
        return decoded[0]