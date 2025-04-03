import os
import tempfile
import traceback
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from google.cloud import storage
import logging

# For Google Drive API
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

def trace_review_scraper_helper():
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
        input_folder_id = os.environ.get("DRIVE_INPUT_FOLDER_ID","1dKkPSGJjnUN_34VqjGZ42JmVB0qOg_WQ")
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