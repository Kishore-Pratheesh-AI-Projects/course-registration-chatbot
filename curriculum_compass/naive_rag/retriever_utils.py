from data_processor import CourseDataProcessor

def load_course_data(file_path: str):
    """Load and process course data.
    Args:
        file_path (str): Path to the course data CSV file 
    Returns:
        list: Processed course data ready for RAG pipeline
    """
    try:
        return CourseDataProcessor.process_course_data(file_path)
    except Exception as e:
        print(f"Error loading course data: {str(e)}")
        return None