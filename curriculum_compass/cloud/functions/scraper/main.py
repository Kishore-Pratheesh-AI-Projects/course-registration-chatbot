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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# from curriculum_compass.cloud.functions.scraper.course_data_scraper import neu_course_scraper_helper
from course_data_scraper import neu_course_scraper_helper
from review_data_loader import trace_review_scraper_helper

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/pratheeshjp/Documents/course-registration-chatbot/curriculum_compass/cloud/curriculum-compass-e59b2e4610e5.json"


# @functions_framework.http
# def scrape_courses_function(request):
#     try:
#         # course_result = neu_course_scraper_helper(request)
#         # trace_result = trace_review_scraper_helper(request)

#         # if not course_result['success'] or not trace_result['success']:
#         #     return {
#         #         "success": False,
#         #         "course_error": course_result.get('error', ''),
#         #         "trace_error": trace_result.get('error', '')
#         #     }
        
#         return {
#             "success": True,
#             "course_file": course_result['file'],
#             # "trace_file": trace_result['file'],
#             "course_record_count": course_result['record_count'],
#             # "trace_record_count": trace_result['record_count']
#         }
#     except Exception as e:
#         logger.error(f"Error in scrape_courses_function: {str(e)}")
#         logger.error(traceback.format_exc())
#         return {"success": False, "error": str(e)}
    


if __name__ == "__main__":

    try :

        course_result = neu_course_scraper_helper()
        trace_result = trace_review_scraper_helper()

        if not course_result['success'] or not trace_result['success']:
            print({
                "success": False,
                "course_error": course_result.get('error', ''),
                "trace_error": trace_result.get('error', '')
            })

        else:
            print({
                "success": True,
                "course_file": course_result['file'],
                "trace_file": trace_result['file'],
                "course_record_count": course_result['record_count'],
                "trace_record_count": trace_result['record_count']
            })

    except Exception as e:

        logger.error(f"Error in scrape_courses_function: {str(e)}")
        logger.error(traceback.format_exc())
        print({"success": False, "error": str(e)})

    