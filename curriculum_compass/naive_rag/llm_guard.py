from typing import Tuple, Dict, Any
import weave
from llm_guard.input_scanners import Gibberish, BanSubstrings, PromptInjection, TokenLimit
from llm_guard.output_scanners import Bias, NoRefusal, Gibberish as OutputGibberish
from llm_guard.scan_output import scan_output
from llm_guard.scan_input import scan_input

class LLMGuard:
    """
    A class to handle input and output content scanning using LLM Guard library.
    """
    def __init__(self, model_name: str):
        """
        Initialize LLM Guard scanners for both input and output.
        
        Args:
            model_name: The name of the model being used (for token limit calculation)
        """
        # Initialize input scanners
        self.input_scanners = {
            "gibberish": Gibberish(),
            "ban_substrings": BanSubstrings(),
            "prompt_injection": PromptInjection(),
            "token_limit": TokenLimit(model_name)
        }
        
        # Initialize output scanners
        self.output_scanners = {
            "bias": Bias(),
            "no_refusal": NoRefusal(),
            "gibberish": OutputGibberish()
        }

    @weave.op(name="scan_input_with_guard")
    def validate_input(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Scan input text using LLM Guard's input scanners.
        
        Args:
            text: Input text to scan
            
        Returns:
            Tuple containing:
            - Boolean indicating if input is safe
            - String containing reason if unsafe
            - Dictionary containing detailed results from each scanner
        """
        try:
            is_valid, results = scan_input(text, self.input_scanners)
            
            # Compile detailed feedback from results
            feedback = ""
            if not is_valid:
                failed_scanners = [
                    scanner_name for scanner_name, result in results.items() 
                    if not result.valid
                ]
                feedback = f"Failed scanners: {', '.join(failed_scanners)}"
            
            return is_valid, feedback, results
            
        except Exception as e:
            return False, f"Error during input scanning: {str(e)}", {}

    @weave.op(name="scan_output_with_guard")
    def validate_output(self, text: str, prompt: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Scan output text using LLM Guard's output scanners.
        
        Args:
            text: Output text to scan
            prompt: Original prompt that generated the output (optional)
            
        Returns:
            Tuple containing:
            - Boolean indicating if output is safe
            - String containing reason if unsafe
            - Dictionary containing detailed results from each scanner
        """
        try:
            is_valid, results = scan_output(text, prompt, self.output_scanners)
            
            # Compile detailed feedback from results
            feedback = ""
            if not is_valid:
                failed_scanners = [
                    scanner_name for scanner_name, result in results.items() 
                    if not result.valid
                ]
                feedback = f"Failed scanners: {', '.join(failed_scanners)}"
            
            return is_valid, feedback, results
            
        except Exception as e:
            return False, f"Error during output scanning: {str(e)}", {}