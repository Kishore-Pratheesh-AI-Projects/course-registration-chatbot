import os
import fitz
import pandas as pd
from typing import Dict, Any
from curriculum_compass.data_pipeline.utils import LoggerConfig, FileManager

class TraceReviewScraper:
    """
    Scraper for extracting course reviews from TRACE PDF files.
    """
    
    def __init__(self, term: str = "(Spring 2024)"):
        """
        Initialize the TRACE review scraper.
        
        Args:
            term (str): Term identifier for filtering course data
        """
        self.logger = LoggerConfig.setup_logging()
        self.TERM = term

    def extract_reviews(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured review data from a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            Dict[str, Any]: Extracted course and review data
        """
        pdf_data = {"questions": {}}

        try:
            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    text = page.get_text()
                    lines = text.splitlines()

                    line_idx = 0
                    while line_idx < len(lines):
                        line = lines[line_idx]
                        
                        # Extract course metadata
                        if self.TERM in line:
                            pdf_data["course_name"] = line
                        elif "Instructor: " in line:
                            pdf_data["instructor"] = line.split("Instructor: ")[1]
                        elif "Subject: " in line:
                            pdf_data["subject"] = line.split("Subject: ")[1]
                        elif "Catalog & Section: " in line:
                            pdf_data["course_number"] = line.split("Catalog & Section: ")[1].split(" ")[0]
                        elif "Course ID: " in line:
                            pdf_data["crn"] = line.split("Course ID: ")[1]
                        
                        # Extract questions and reviews
                        elif "Q: " in line:
                            question = line.split("Q: ")[-1]
                            reviews = []

                            line_idx += 1
                            while line_idx < len(lines):
                                if lines[line_idx].isnumeric():
                                    skip_idx = 1
                                    while line_idx + skip_idx < len(lines) and lines[line_idx + skip_idx] == "":
                                        skip_idx += 1
                                    
                                    if line_idx + skip_idx < len(lines):
                                        reviews.append(lines[line_idx + skip_idx])

                                    line_idx += skip_idx
                                elif "Q: " in lines[line_idx]:
                                    break
                                else:
                                    line_idx += 1

                            pdf_data["questions"][question] = reviews
                            continue
                        
                        line_idx += 1
        
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return pdf_data

    def process_reviews(self, data_directory: str) -> pd.DataFrame:
        """
        Process all review PDFs in the given directory.
        
        Args:
            data_directory (str): Path to the directory containing review PDFs
        
        Returns:
            pd.DataFrame: DataFrame with consolidated review data
        """
        self.logger.info(f"Processing reviews from directory: {data_directory}")
        
        # Ensure directory exists
        FileManager.ensure_directory(data_directory)
        
        # Collect review data
        reviews_data = []
        
        # Process each PDF file
        for filename in os.listdir(data_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(data_directory, filename)
                crn = os.path.splitext(filename)[0]
                
                # Extract reviews from PDF
                pdf_reviews = self.extract_reviews(pdf_path)
                
                # Process extracted reviews
                for question, review_list in pdf_reviews.get('questions', {}).items():
                    for review in review_list:
                        reviews_data.append({
                            'CRN': crn,
                            'Course Name': pdf_reviews.get('course_name', ''),
                            'Instructor': pdf_reviews.get('instructor', ''),
                            'Subject': pdf_reviews.get('subject', ''),
                            'Course Number': pdf_reviews.get('course_number', ''),
                            'Question': question,
                            'Review': review
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews_data)
        self.logger.info(f"Processed {len(df)} reviews")
        
        return df

