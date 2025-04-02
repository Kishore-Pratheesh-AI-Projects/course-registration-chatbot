import functions_framework
import os
import tempfile
from google.cloud import storage
import traceback
import pandas as pd
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_time(time_val):
    """Clean and format time values."""
    if pd.isna(time_val) or time_val == '' or time_val == 0:
        return None
    return str(int(time_val)).zfill(4)

def format_time(time_str):
    """Format time string to AM/PM format."""
    if not time_str or len(time_str) != 4:
        return None
    hours = int(time_str[:2])
    minutes = time_str[2:]
    period = "AM" if hours < 12 else "PM"
    if hours > 12:
        hours -= 12
    elif hours == 0:
        hours = 12
    return f"{hours}:{minutes} {period}"

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

@functions_framework.cloud_event
def process_data_function(cloud_event):
    """Cloud Function triggered by a Cloud Storage event."""
    try:
        # Get storage client
        storage_client = storage.Client()
        
        # Get bucket and file info from event
        bucket_name = cloud_event.data["bucket"]
        file_name = cloud_event.data["name"]
        
        # Process only course files
        if not (file_name.startswith('courses/') and file_name.endswith('.csv')):
            logger.info(f"Skipping non-course file: {file_name}")
            return {"success": True, "message": "Skipped non-course file"}
        
        # Download file from GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file_path = temp_file.name
        
        # Read CSV into DataFrame
        df = pd.read_csv(temp_file_path)
        os.unlink(temp_file_path)  # Clean up temp file
        
        # Process data
        logger.info(f"Processing course data from {file_name}")
        processed_courses = [course_to_structured_text(row) for _, row in df.iterrows()]
        
        # Save processed data
        output_bucket_name = os.environ.get('OUTPUT_BUCKET', 'curriculum-compass-processed-data')
        output_bucket = storage_client.bucket(output_bucket_name)
        
        # Create a unique output filename based on the input
        base_name = os.path.basename(file_name)
        output_file_name = f"processed/{base_name.replace('.csv', '')}_processed.jsonl"
        output_blob = output_bucket.blob(output_file_name)
        
        # Save as JSONL (one JSON per line)
        with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w+', delete=False) as temp_output:
            for i, course_text in enumerate(processed_courses):
                temp_output.write(json.dumps({
                    "id": f"course_{i}",
                    "content": course_text
                }) + '\n')
            temp_output_path = temp_output.name
        
        # Upload to GCS
        output_blob.upload_from_filename(temp_output_path)
        
        # Also save a "latest" version
        if 'latest' in file_name:
            latest_output_name = "processed/courses_processed_latest.jsonl"
            latest_output_blob = output_bucket.blob(latest_output_name)
            latest_output_blob.upload_from_filename(temp_output_path)
        
        os.unlink(temp_output_path)  # Clean up temp file
        
        logger.info(f"Processed data uploaded to gs://{output_bucket_name}/{output_file_name}")
        
        return {
            "success": True, 
            "input_file": file_name, 
            "output_file": output_file_name,
            "course_count": len(processed_courses)
        }
        
    except Exception as e:
        logger.error(f"Error in data processor function: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}