from curriculum_compass.data_pipeline.neu_course_scraper import NEUCourseScraper
from curriculum_compass.data_pipeline.trace_review_scraper import TraceReviewScraper
from curriculum_compass.data_pipeline.utils import FileManager

def main():

    FileManager.ensure_directory('notebooks/data')

    print("\n=== Scraping Northeastern Courses ===")
    course_scraper = NEUCourseScraper()
    courses_df = course_scraper.scrape_courses(subject="CS")
    FileManager.save_dataframe(courses_df, 'notebooks/data/courses.csv')

    
    print("\n=== Processing TRACE Reviews ===")
    review_scraper = TraceReviewScraper()
    reviews_df = review_scraper.process_reviews('notebooks/data')
    FileManager.save_dataframe(reviews_df, 'notebooks/data/reviews.csv')

if __name__ == "__main__":
    main()