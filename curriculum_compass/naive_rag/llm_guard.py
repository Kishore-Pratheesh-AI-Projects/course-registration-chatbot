from typing import Tuple, Dict, Any
import weave
from llm_guard.input_scanners import Gibberish, BanSubstrings, PromptInjection, TokenLimit
from llm_guard.output_scanners import Bias, NoRefusal, Gibberish as OutputGibberish

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
        self.input_scanners = [
            Gibberish(),
            BanSubstrings(),
            PromptInjection(),
            TokenLimit(model_name)
        ]
        
        # Initialize output scanners
        self.output_scanners = [
            Bias(),
            NoRefusal(),
            OutputGibberish()
        ]

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
            results = [scanner.is_valid(text) for scanner in self.input_scanners]
            is_valid = all(result.is_valid for result in results)
            
            # Compile detailed feedback from results
            feedback = ""
            if not is_valid:
                failed_scanners = [
                    type(scanner).__name__ for scanner, result in zip(self.input_scanners, results)
                    if not result.is_valid
                ]
                feedback = f"Failed scanners: {', '.join(failed_scanners)}"
            
            # Convert results to dictionary for consistency
            results_dict = {
                type(scanner).__name__: {
                    'is_valid': result.is_valid,
                    'risk_score': result.risk_score,
                    'comment': result.comment
                }
                for scanner, result in zip(self.input_scanners, results)
            }
            
            return is_valid, feedback, results_dict
            
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
            results = [scanner.is_valid(text, prompt) for scanner in self.output_scanners]
            is_valid = all(result.is_valid for result in results)
            
            # Compile detailed feedback from results
            feedback = ""
            if not is_valid:
                failed_scanners = [
                    type(scanner).__name__ for scanner, result in zip(self.output_scanners, results)
                    if not result.is_valid
                ]
                feedback = f"Failed scanners: {', '.join(failed_scanners)}"
            
            # Convert results to dictionary for consistency
            results_dict = {
                type(scanner).__name__: {
                    'is_valid': result.is_valid,
                    'risk_score': result.risk_score,
                    'comment': result.comment
                }
                for scanner, result in zip(self.output_scanners, results)
            }
            
            return is_valid, feedback, results_dict
            
        except Exception as e:
            return False, f"Error during output scanning: {str(e)}", {}