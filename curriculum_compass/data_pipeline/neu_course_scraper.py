import requests
import pandas as pd
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from curriculum_compass.data_pipeline.utils import LoggerConfig, RequestHandler

class NEUCourseScraper:
    def __init__(self, base_url: str = "https://nubanner.neu.edu/StudentRegistrationSsb/ssb"):
        """
        Initialize NEU Course Scraper.
        
        Args:
            base_url (str): Base URL for course registration system
        """
        self.logger = LoggerConfig.setup_logging()
        self.base_url = base_url
        self.term = "202530" # Spring 2025 term code

    def get_course_list(self, cookies: Dict[str, str], subject: str = "CS") -> List[Dict[str, Any]]:
        """
        Retrieve list of courses for a specific subject.
        
        Args:
            cookies (Dict[str, str]): Session cookies
            subject (str): Course subject code
        
        Returns:
            List[Dict[str, Any]]: List of course details
        """
        url = f"{self.base_url}/searchResults/searchResults"
        headers = RequestHandler.prepare_cookie_header(cookies)
        
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
        """
        Extract detailed information for a specific course.
        
        Args:
            cookies (Dict[str, str]): Session cookies
            course (Dict[str, Any]): Course basic information
        
        Returns:
            Dict[str, Any]: Detailed course information
        """
        crn = course.get('courseReferenceNumber')
        headers = RequestHandler.prepare_cookie_header(cookies)
        
        details = {
            'CRN': crn,
            'Course Title': course.get('courseTitle', ''),
            'Subject Course': course.get('subjectCourse', ''),
            'Campus': course.get('campusDescription', '')
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
        """
        Main method to scrape courses for a given subject.
        
        Args:
            subject (str): Subject code to scrape
        
        Returns:
            pd.DataFrame: DataFrame with course details
        """
        # Get session cookies
        cookies = RequestHandler.get_session_cookies(self.base_url)
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

# def main():
#     """
#     Main execution function for course scraping.
#     """
#     FileManager.ensure_directory('./data')
    
#     scraper = NEUCourseScraper()
#     df = scraper.scrape_courses()
    
#     if not df.empty:
#         FileManager.save_dataframe(df, './data/courses.csv')
#     else:
#         print("No courses were scraped.")

# if __name__ == "__main__":
#     main()