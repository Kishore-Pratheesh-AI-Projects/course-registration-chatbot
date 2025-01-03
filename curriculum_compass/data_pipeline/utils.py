import os
import logging
import requests
from typing import Dict, Any, List

class LoggerConfig:
    @staticmethod
    def setup_logging(log_level: int = logging.INFO):
        """
        Set up standardized logging configuration.
        
        Args:
            log_level (int): Logging level
        
        Returns:
            logging.Logger: Configured logger
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

class RequestHandler:
    @staticmethod
    def get_session_cookies(base_url: str, term: str = "202530") -> Dict[str, str]:
        """
        Fetch session cookies for a given base URL and term.
        
        Args:
            base_url (str): Base URL for requests
            term (str): Term identifier
        
        Returns:
            Dict[str, str]: Cookies dictionary
        """
        logger = LoggerConfig.setup_logging()
        
        try:
            url = f"{base_url}/term/search"
            headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
            body = {"term": term}
            
            logger.info("Attempting to get session cookies...")
            response = requests.post(url, headers=headers, data=body)
            
            if response.ok:
                logger.info("Successfully obtained cookies")
                return response.cookies.get_dict()
            else:
                logger.error(f"Failed to get cookies. Status code: {response.status_code}")
                return {}
        
        except Exception as e:
            logger.error(f"Error getting cookies: {e}")
            return {}

    @staticmethod
    def prepare_cookie_header(cookies: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare cookie header from cookies dictionary.
        
        Args:
            cookies (Dict[str, str]): Cookies dictionary
        
        Returns:
            Dict[str, str]: Headers with cookie string
        """
        return {"Cookie": "; ".join([f"{key}={value}" for key, value in cookies.items()])}

class FileManager:
    @staticmethod
    def ensure_directory(directory_path: str) -> None:
        """
        Ensure the specified directory exists.
        
        Args:
            directory_path (str): Path to the directory
        """
        os.makedirs(directory_path, exist_ok=True)

    @staticmethod
    def save_dataframe(df, output_path: str) -> None:
        """
        Save DataFrame to a CSV file.
        
        Args:
            df: Pandas DataFrame
            output_path (str): Path to save the CSV
        """
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")