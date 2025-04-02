# # TODO: Implement the data fetching logic for Trace reviews ansd conve
# import functions_framework
# import os
# import tempfile
# from google.cloud import storage
# import traceback
# import json
# import pandas as pd
# import requests
# from typing import Dict, Any, List
# from concurrent.futures import ThreadPoolExecutor
# from functools import partial
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class NEUCourseScraper:
#     def __init__(self, base_url: str = "https://nubanner.neu.edu/StudentRegistrationSsb/ssb"):
#         """Initialize NEU Course Scraper."""
#         self.logger = logger
#         self.base_url = base_url
#         self.term = "202530"  # Spring 2025 term code

#     def prepare_cookie_header(self, cookies: Dict[str, str]) -> Dict[str, str]:
#         """Prepare cookie header from cookies dictionary."""
#         return {"Cookie": "; ".join([f"{key}={value}" for key, value in cookies.items()])}

#     def get_session_cookies(self) -> Dict[str, str]:
#         """Fetch session cookies."""
#         try:
#             url = f"{self.base_url}/term/search"
#             headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
#             body = {"term": self.term}
            
#             self.logger.info("Attempting to get session cookies...")
#             response = requests.post(url, headers=headers, data=body)
            
#             if response.ok:
#                 self.logger.info("Successfully obtained cookies")
#                 return response.cookies.get_dict()
#             else:
#                 self.logger.error(f"Failed to get cookies. Status code: {response.status_code}")
#                 return {}
        
#         except Exception as e:
#             self.logger.error(f"Error getting cookies: {e}")
#             return {}

#     def get_course_list(self, cookies: Dict[str, str], subject: str = "CS") -> List[Dict[str, Any]]:
#         """Retrieve list of courses for a specific subject."""
#         url = f"{self.base_url}/searchResults/searchResults"
#         headers = self.prepare_cookie_header(cookies)
        
#         params = {
#             "txt_subject": subject,
#             "txt_term": self.term,
#             "pageOffset": 0,
#             "pageMaxSize": 100000000
#         }
        
#         try:
#             response = requests.get(url, headers=headers, params=params)
#             response.raise_for_status()
#             return response.json().get('data', [])
#         except Exception as e:
#             self.logger.error(f"Error fetching course list: {e}")
#             return []

#     def get_course_details(self, cookies: Dict[str, str], course: Dict[str, Any]) -> Dict[str, Any]:
#         """Extract detailed information for a specific course."""
#         crn = course.get('courseReferenceNumber')
#         headers = self.prepare_cookie_header(cookies)
        
#         details = {
#             'CRN': crn,
#             'Course Title': course.get('courseTitle', ''),
#             'Subject Course': course.get('subjectCourse', ''),
#             'Campus Description': course.get('campusDescription', ''),
#             'Subject': course.get('subject', ''),
#             'Course Number': course.get('courseNumber', ''),
#             'Term': self.term,
#             'Course Description': course.get('courseDescription', ''),
#             'Prerequisites': course.get('prerequisites', '[]')
#         }
        
#         # Fetch faculty info
#         try:
#             faculty_url = f"{self.base_url}/searchResults/getFacultyMeetingTimes"
#             faculty_params = {"term": self.term, "courseReferenceNumber": crn}
#             faculty_response = requests.get(faculty_url, headers=headers, params=faculty_params)
            
#             if faculty_response.ok:
#                 faculty_data = faculty_response.json().get("fmt", [{}])[0]
#                 meeting_time = faculty_data.get("meetingTime", {})
#                 faculty = faculty_data.get("faculty", [{}])[0]
                
#                 details.update({
#                     'Faculty Name': faculty.get('displayName', ''),
#                     'Begin Time': meeting_time.get('beginTime', ''),
#                     'End Time': meeting_time.get('endTime', ''),
#                     'Days': ', '.join([
#                         day.capitalize() 
#                         for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] 
#                         if meeting_time.get(day)
#                     ])
#                 })
#         except Exception as e:
#             self.logger.error(f"Error fetching faculty info for {crn}: {e}")
        
#         return details

#     def scrape_courses(self, subject: str = "CS") -> pd.DataFrame:
#         """Main method to scrape courses for a given subject."""
#         # Get session cookies
#         cookies = self.get_session_cookies()
#         if not cookies:
#             return pd.DataFrame()

#         # Get course list
#         courses = self.get_course_list(cookies, subject)
#         self.logger.info(f"Found {len(courses)} courses")

#         # Process courses in parallel
#         course_details = []
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             detailed_courses = list(executor.map(
#                 partial(self.get_course_details, cookies), 
#                 courses
#             ))
#             course_details.extend(detailed_courses)

#         # Create DataFrame
#         df = pd.DataFrame(course_details)
#         return df
    
# def neu_course_scraper_helper(request):
#     """HTTP Cloud Function that scrapes course data and uploads to GCS."""
#     try:
#         storage_client = storage.Client()
#         bucket_name = os.environ.get('COURSE_BUCKET_RAW_DATA', 'curriculum-compass-neu-banner-data')
        
#         # Get subject from request or use default
#         request_json = request.get_json(silent=True)
#         subject = "CS"
#         if request_json and 'subject' in request_json:
#             subject = request_json['subject']
            
#         # Scrape courses
#         logger.info(f"Starting to scrape {subject} courses")
#         scraper = NEUCourseScraper()
#         df = scraper.scrape_courses(subject)
        
#         if df.empty:
#             return {"success": False, "error": "No courses were scraped."}
        
#         # Create a temporary file to save the data
#         with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as temp_file:
#             df.to_csv(temp_file.name, index=False)
#             temp_file_path = temp_file.name
            
#         # Upload to GCS
#         bucket = storage_client.bucket(bucket_name)
#         timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
#         destination_blob_name = f"courses/{subject.lower()}_courses_{timestamp}.csv"
#         blob = bucket.blob(destination_blob_name)
        
#         blob.upload_from_filename(temp_file_path)
        
#         # Also upload a "latest" version that will overwrite previous versions
#         latest_blob_name = f"courses/{subject.lower()}_courses_latest.csv"
#         latest_blob = bucket.blob(latest_blob_name)
#         latest_blob.upload_from_filename(temp_file_path)
        
#         os.unlink(temp_file_path)  # Clean up temp file
        
#         logger.info(f"Course data uploaded to gs://{bucket_name}/{destination_blob_name}")
        
#         return {"success": True, "file": destination_blob_name, "record_count": len(df)}
        
#     except Exception as e:
#         logger.error(f"Error in course scraper function: {str(e)}")
#         logger.error(traceback.format_exc())
#         return {"success": False, "error": str(e)}
    

# class TraceReviewScraper:
    
# def trace_review_scraper_helper(request):
#     """HTTP Cloud Function that scrapes TRACE reviews and uploads to GCS."""
#     try:
#         pass
        
        
#     except Exception as e:
#         pass
       


# @functions_framework.http
# def scrape_courses_function(request):
#     try:
#         course_result = neu_course_scraper_helper(request)
#         trace_result = trace_review_scraper_helper(request)

#         if not course_result['success'] or not trace_result['success']:
#             return {
#                 "success": False,
#                 "course_error": course_result.get('error', ''),
#                 "trace_error": trace_result.get('error', '')
#             }
        
#         return {
#             "success": True,
#             "course_file": course_result['file'],
#             "trace_file": trace_result['file'],
#             "course_record_count": course_result['record_count'],
#             "trace_record_count": trace_result['record_count']
#         }
#     except Exception as e:
#         logger.error(f"Error in scrape_courses_function: {str(e)}")
#         logger.error(traceback.format_exc())
#         return {"success": False, "error": str(e)}
    


# # TODO: Make that term code for neu banner data dynamic based on the current term, include google API in the GCP service account


import functions_framework
import os
import tempfile
import traceback
import json
import pandas as pd
import requests
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import fitz  # PyMuPDF for PDF processing
from google.cloud import storage

# For Google Drive API
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


############################################
# NEU Course Scraper and Helper Functions
############################################

class NEUCourseScraper:
    def __init__(self, base_url: str = "https://nubanner.neu.edu/StudentRegistrationSsb/ssb"):
        """Initialize NEU Course Scraper."""
        self.logger = logger
        self.base_url = base_url
        self.term = "202530"  # Spring 2025 term code 

    def prepare_cookie_header(self, cookies: Dict[str, str]) -> Dict[str, str]:
        """Prepare cookie header from cookies dictionary."""
        return {"Cookie": "; ".join([f"{key}={value}" for key, value in cookies.items()])}

    def get_session_cookies(self) -> Dict[str, str]:
        """Fetch session cookies."""
        try:
            url = f"{self.base_url}/term/search"
            headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
            body = {"term": self.term}
            
            self.logger.info("Attempting to get session cookies...")
            response = requests.post(url, headers=headers, data=body)
            
            if response.ok:
                self.logger.info("Successfully obtained cookies")
                return response.cookies.get_dict()
            else:
                self.logger.error(f"Failed to get cookies. Status code: {response.status_code}")
                return {}
        
        except Exception as e:
            self.logger.error(f"Error getting cookies: {e}")
            return {}

    def get_course_list(self, cookies: Dict[str, str], subject: str = "CS") -> List[Dict[str, Any]]:
        """Retrieve list of courses for a specific subject."""
        url = f"{self.base_url}/searchResults/searchResults"
        headers = self.prepare_cookie_header(cookies)
        
        params = {
            "txt_subject": subject,
            "txt_term": self.term,
            "pageOffset": 0,
            "pageMaxSize": 100000000
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('data', [])
        except Exception as e:
            self.logger.error(f"Error fetching course list: {e}")
            return []

    def get_course_details(self, cookies: Dict[str, str], course: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed information for a specific course."""
        crn = course.get('courseReferenceNumber')
        headers = self.prepare_cookie_header(cookies)
        
        details = {
            'CRN': crn,
            'Course Title': course.get('courseTitle', ''),
            'Subject Course': course.get('subjectCourse', ''),
            'Campus Description': course.get('campusDescription', ''),
            'Subject': course.get('subject', ''),
            'Course Number': course.get('courseNumber', ''),
            'Term': self.term,
            'Course Description': course.get('courseDescription', ''),
            'Prerequisites': course.get('prerequisites', '[]')
        }
        
        # Fetch faculty info
        try:
            faculty_url = f"{self.base_url}/searchResults/getFacultyMeetingTimes"
            faculty_params = {"term": self.term, "courseReferenceNumber": crn}
            faculty_response = requests.get(faculty_url, headers=headers, params=faculty_params)
            
            if faculty_response.ok:
                faculty_data = faculty_response.json().get("fmt", [{}])[0]
                meeting_time = faculty_data.get("meetingTime", {})
                faculty = faculty_data.get("faculty", [{}])[0]
                
                details.update({
                    'Faculty Name': faculty.get('displayName', ''),
                    'Begin Time': meeting_time.get('beginTime', ''),
                    'End Time': meeting_time.get('endTime', ''),
                    'Days': ', '.join([
                        day.capitalize() 
                        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] 
                        if meeting_time.get(day)
                    ])
                })
        except Exception as e:
            self.logger.error(f"Error fetching faculty info for {crn}: {e}")
        
        return details

    def scrape_courses(self, subject: str = "CS") -> pd.DataFrame:
        """Main method to scrape courses for a given subject."""
        # Get session cookies
        cookies = self.get_session_cookies()
        if not cookies:
            return pd.DataFrame()

        # Get course list
        courses = self.get_course_list(cookies, subject)
        self.logger.info(f"Found {len(courses)} courses")

        # Process courses in parallel
        course_details = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            detailed_courses = list(executor.map(
                partial(self.get_course_details, cookies), 
                courses
            ))
            course_details.extend(detailed_courses)

        # Create DataFrame
        df = pd.DataFrame(course_details)
        return df

def neu_course_scraper_helper(request):
    """HTTP Cloud Function that scrapes course data and uploads to GCS."""
    try:
        storage_client = storage.Client()
        bucket_name = os.environ.get('COURSE_BUCKET_RAW_DATA', 'curriculum-compass-neu-banner-data')
        
        # Use "CS" as default subject
        subject = "CS"
            
        # Scrape courses
        logger.info(f"Starting to scrape {subject} courses")
        scraper = NEUCourseScraper()
        df = scraper.scrape_courses(subject)
        
        if df.empty:
            return {"success": False, "error": "No courses were scraped."}
        
        # Create a temporary file to save the data
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
            
        # Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        destination_blob_name = f"courses/{subject.lower()}_courses_{timestamp}.csv"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(temp_file_path)
        
        # Also upload a "latest" version that will overwrite previous versions
        latest_blob_name = f"courses/{subject.lower()}_courses_latest.csv"
        latest_blob = bucket.blob(latest_blob_name)
        latest_blob.upload_from_filename(temp_file_path)
        
        os.unlink(temp_file_path)  # Clean up temp file
        
        logger.info(f"Course data uploaded to gs://{bucket_name}/{destination_blob_name}")
        
        return {"success": True, "file": destination_blob_name, "record_count": len(df)}
        
    except Exception as e:
        logger.error(f"Error in course scraper function: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


############################################
# TRACE Reviews Data Scraper and Helper (Drive Input, GCS Output)
############################################

class TraceDataScraper:
    def extract_questions_and_reviews_from_pdf(self, pdf_path):
        """
        Extracts structured data, including questions and their reviews, from a PDF.

        Parameters:
            pdf_path (str): The file path of the PDF file.

        Returns:
            dict: A dictionary containing extracted data such as course details and questions with reviews.
        """
        TERM = "(Spring 2024)"  # Term identifier to filter course-specific data
        pdf_data = {"questions": {}}

        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text()
                    lines = text.splitlines()

                    line_idx = 0
                    while line_idx < len(lines):
                        line = lines[line_idx]

                        if TERM in line:
                            pdf_data["course_name"] = line

                        elif "Instructor: " in line:
                            pdf_data["instructor"] = line.split("Instructor: ")[1]

                        elif "Subject: " in line:
                            pdf_data["subject"] = line.split("Subject: ")[1]

                        elif "Catalog & Section: " in line:
                            section_part = line.split("Catalog & Section: ")[1]
                            pdf_data["course_number"] = section_part.split(" ")[0]

                        elif "Course ID: " in line:
                            pdf_data["crn"] = line.split("Course ID: ")[1]

                        elif "Q: " in line:
                            question = line.split("Q: ")[-1]
                            reviews = []

                            line_idx += 1
                            while line_idx < len(lines):
                                if lines[line_idx].isnumeric():
                                    skip_idx = 1
                                    while (line_idx + skip_idx) < len(lines) and lines[line_idx + skip_idx] == "":
                                        skip_idx += 1
                                    if (line_idx + skip_idx) < len(lines):
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
            logger.error(f"An error occurred while processing the PDF {pdf_path}: {e}")

        return pdf_data

    def process_directory(self, data_directory):
        """
        Processes all PDF files in the given directory and consolidates review information.

        Parameters:
            data_directory (str): The path to the directory containing PDF files.

        Returns:
            dict: A dictionary containing consolidated data for all PDF files.
        """
        logger.info(f"Starting to process directory: {data_directory}")
        consolidated_data = {}

        try:
            for filename in os.listdir(data_directory):
                if filename.endswith(".pdf"):
                    crn = filename.replace(".pdf", "")
                    pdf_path = os.path.join(data_directory, filename)
                    pdf_data = self.extract_questions_and_reviews_from_pdf(pdf_path)
                    consolidated_data[crn] = pdf_data

        except Exception as e:
            logger.error(f"An error occurred while processing the directory {data_directory}: {e}")

        logger.info(f"Completed processing of directory: {data_directory}")
        return consolidated_data

def trace_review_scraper_helper(request):
    """
    HTTP Cloud Function that:
      1. Uses the Google Drive API to list and download PDF files from a specified Drive folder.
      2. Processes the downloaded PDFs using TraceDataScraper.
      3. Flattens the consolidated data into a CSV.
      4. Uploads the CSV to a Google Cloud Storage bucket (TRACE_BUCKET_RAW_DATA).
      
    Required environment variables:
      - DRIVE_INPUT_FOLDER_ID: The ID of the Drive folder containing the PDFs.
      - TRACE_BUCKET_RAW_DATA: The name of the GCS bucket where the CSV will be stored.
    """
    try:
        # Set up Drive API credentials and service
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        creds, _ = google.auth.default(scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=creds)

        # Retrieve the Drive folder ID from environment variables
        input_folder_id = os.environ.get("DRIVE_INPUT_FOLDER_ID")
        if not input_folder_id:
            msg = "DRIVE_INPUT_FOLDER_ID environment variable not set."
            logger.error(msg)
            return {"success": False, "error": msg}

        # Query to list all PDFs in the Drive folder
        query = f"'{input_folder_id}' in parents and mimeType='application/pdf'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            msg = "No PDF files found in the specified Google Drive folder."
            logger.warning(msg)
            return {"success": False, "error": msg}

        # Create a local directory for downloaded PDFs
        local_pdf_dir = "/tmp/drive_pdfs"
        if not os.path.exists(local_pdf_dir):
            os.makedirs(local_pdf_dir)

        scraper = TraceDataScraper()
        pdf_count = 0

        # Download each PDF from Drive
        for item in items:
            file_id = item['id']
            file_name = item['name']
            if not file_name.lower().endswith(".pdf"):
                continue

            pdf_count += 1
            local_pdf_path = os.path.join(local_pdf_dir, file_name)
            logger.info(f"Downloading {file_name} from Drive to {local_pdf_path}")

            request_drive = drive_service.files().get_media(fileId=file_id)
            with open(local_pdf_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request_drive)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"Download {file_name}: {int(status.progress() * 100)}%.")

        if pdf_count == 0:
            msg = "No PDF files were downloaded from Drive."
            logger.warning(msg)
            return {"success": False, "error": msg}

        # Process the downloaded PDFs using TraceDataScraper
        consolidated_data = scraper.process_directory(local_pdf_dir)

        # Flatten the consolidated data into rows for a DataFrame
        rows = []
        for crn, data_dict in consolidated_data.items():
            course_name = data_dict.get("course_name", "")
            instructor = data_dict.get("instructor", "")
            subject = data_dict.get("subject", "")
            course_number = data_dict.get("course_number", "")
            questions = data_dict.get("questions", {})

            for question, reviews in questions.items():
                for review in reviews:
                    rows.append({
                        "crn": crn,
                        "course_name": course_name,
                        "instructor": instructor,
                        "subject": subject,
                        "course_number": course_number,
                        "question": question,
                        "review": review
                    })

        if not rows:
            msg = "Parsed PDFs but found no question/review data."
            logger.warning(msg)
            return {"success": False, "error": msg}

        df = pd.DataFrame(rows)
        record_count = len(df)

        # Write the DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        # Upload the CSV file to GCS
        storage_client = storage.Client()
        bucket_name = os.environ.get("TRACE_BUCKET_RAW_DATA", "curriculum-compass-trace-review-data")
        bucket = storage_client.bucket(bucket_name)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        destination_blob_name = f"reviews/trace_reviews_{timestamp}.csv"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(temp_file_path)

        # Also upload a "latest" version that overwrites previous versions
        latest_blob_name = "reviews/trace_reviews_latest.csv"
        latest_blob = bucket.blob(latest_blob_name)
        latest_blob.upload_from_filename(temp_file_path)

        os.unlink(temp_file_path)  # Clean up temporary CSV file

        logger.info(f"TRACE review data uploaded to gs://{bucket_name}/{destination_blob_name}")

        return {
            "success": True,
            "file": destination_blob_name,
            "record_count": record_count
        }

    except Exception as e:
        logger.error(f"Error in trace_review_scraper_helper: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


############################################
# Main Cloud Function Entry Point
############################################

@functions_framework.http
def scrape_courses_function(request):
    try:
        course_result = neu_course_scraper_helper(request)
        trace_result = trace_review_scraper_helper(request)

        if not course_result['success'] or not trace_result['success']:
            return {
                "success": False,
                "course_error": course_result.get('error', ''),
                "trace_error": trace_result.get('error', '')
            }
        
        return {
            "success": True,
            "course_file": course_result['file'],
            "trace_file": trace_result['file'],
            "course_record_count": course_result['record_count'],
            "trace_record_count": trace_result['record_count']
        }
    except Exception as e:
        logger.error(f"Error in scrape_courses_function: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}