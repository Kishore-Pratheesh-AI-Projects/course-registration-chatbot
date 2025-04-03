
import os
import tempfile
import traceback
import pandas as pd
import requests
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from google.cloud import storage


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



def neu_course_scraper_helper():
    """HTTP Cloud Function that scrapes course data and uploads to GCS."""
    try:
        storage_client = storage.Client()
        # bucket_name = os.environ.get('COURSE_BUCKET_RAW_DATA', 'curriculum-compass-neu-banner-data')
        bucket_name = "curriculum-compass-raw-data"
        
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