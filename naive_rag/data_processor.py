import pandas as pd
from .utils import clean_time, format_time

class CourseDataProcessor:
    @staticmethod
    def course_to_structured_text(row):
        """Convert course data to structured, searchable format"""
        metadata_section = (
            "=== COURSE METADATA ===\n"
            f"Course Code: {row.get('Subject Course', '')}\n"
            f"CRN: {str(row.get('CRN', ''))}\n"
            f"Title: {row.get('Course Title', '')}\n"
        )

        campus = row.get('Campus Description', '')
        format_type = ("Online" if campus.lower() == 'online' 
                      else "Self-paced" if campus.lower() == 'no campus, no room needed' 
                      else "In-Person")
        location_section = (
            "=== LOCATION ===\n"
            f"Campus: {campus}\n"
            f"Format: {format_type}\n"
        )

        begin_time = clean_time(row.get('Begin Time', ''))
        end_time = clean_time(row.get('End Time', ''))
        days = row.get('Days', '')
        schedule_section = "=== SCHEDULE ===\n"
        if begin_time and end_time and days:
            schedule_section += (f"Days: {days}\n"
                               f"Time: {format_time(begin_time)} to {format_time(end_time)}\n")
        else:
            schedule_section += "Schedule: Flexible/Self-paced\n"

        faculty = row.get('Faculty Name', '')
        instructor_section = (
            "=== INSTRUCTOR ===\n"
            f"Professor: {faculty if faculty else 'Not specified'}\n"
        )

        prerequisites = row.get('Prerequisites', '[]')
        prereq_text = ("None required" if prerequisites == '[]' or not prerequisites or prerequisites.strip() == '' 
                      else prerequisites.strip('[]').replace("'", "").replace('"', ''))
        details_section = (
            "=== COURSE DETAILS ===\n"
            f"Term: {row.get('Term', '')}\n"
            f"Prerequisites: {prereq_text}\n"
        )

        description = row.get('Course Description', '')
        description_section = (
            "=== DESCRIPTION ===\n"
            f"{description if description else 'No description available'}\n"
        )

        return (f"{metadata_section.lower()}\n"
                f"{location_section.lower()}\n"
                f"{schedule_section.lower()}\n"
                f"{instructor_section.lower()}\n"
                f"{details_section.lower()}\n"
                f"{description_section.lower()}")

    @staticmethod
    def process_course_data(file_path):
        """Process entire course dataset and convert to structured text."""
        df = pd.read_csv(file_path)
        return [CourseDataProcessor.course_to_structured_text(row) for _, row in df.iterrows()]