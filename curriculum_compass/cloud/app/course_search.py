import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import storage
import json
from config import get_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudCourseSearchSystem:
    """
    TF-IDF based search system for course data, adapted for GCP
    """
    def __init__(self):
        self.documents = None
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b|===\s*\w+\s*===',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.storage_client = storage.Client()
        self.config = get_config()

    def preprocess_query(self, query):
        """Extract structured information from query"""
        query_parts = {
            'course': None,
            'professor': None,
            'term': None,
            'campus': None
        }
        
        query = query.lower()
        
        # Extract course information (extend based on your needs)
        if 'algorithms' in query:
            query_parts['course'] = 'algorithms'
        elif 'artificial intelligence' in query:
            query_parts['course'] = 'artificial intelligence'
            
        # Extract professor name
        if 'rajagopal' in query or 'venkatesaramani' in query:
            query_parts['professor'] = 'venkatesaramani, rajagopal'
            
        # Extract term
        if 'spring 2025' in query:
            query_parts['term'] = 'spring 2025'
            
        # Extract campus
        if 'boston' in query:
            query_parts['campus'] = 'boston'
            
        return query_parts

    def enhance_query(self, query):
        """Enhance query with structural information"""
        query_parts = self.preprocess_query(query)
        enhanced_query = query.lower()
        
        if query_parts['course']:
            enhanced_query += f" === course metadata === title: {query_parts['course']}"
        if query_parts['professor']:
            enhanced_query += f" === instructor === professor: {query_parts['professor']}"
        if query_parts['term']:
            enhanced_query += f" === course details === term: {query_parts['term']}"
        if query_parts['campus']:
            enhanced_query += f" === location === campus: {query_parts['campus']}"
            
        return enhanced_query, query_parts

    def load_course_data_from_gcs(self):
        """Load course data from Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.config.PROCESSED_DATA_BUCKET)
            blob = bucket.blob("processed/courses_processed_latest.jsonl")
            
            # Download as string
            content = blob.download_as_text()
            
            # Parse JSONL
            import json
            course_data = []
            for line in content.splitlines():
                if line.strip():
                    record = json.loads(line)
                    course_data.append(record.get('content', ''))
            
            logger.info(f"Loaded {len(course_data)} course documents from Cloud Storage")
            return course_data
            
        except Exception as e:
            logger.error(f"Error loading course data from GCS: {e}")
            return []

    def add_course_sentences_to_db(self, course_data=None):
        """Add processed course data to the search system"""
        # If no data provided, load from GCS
        if course_data is None:
            course_data = self.load_course_data_from_gcs()
            
        if not course_data:
            logger.warning("No course data to index")
            return False
            
        self.documents = [doc for doc in course_data if doc is not None]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        logger.info(f"Indexed {len(self.documents)} course documents with TF-IDF")
        return True

    def query_courses(self, query_text, n_results=5):
        """Query courses based on enhanced query"""
        enhanced_query, query_parts = self.enhance_query(query_text)
        
        try:
            if self.tfidf_matrix is None:
                logger.warning("TF-IDF matrix not initialized, returning empty results")
                return {"documents": [["No documents indexed"]]}
            
            query_vec = self.vectorizer.transform([enhanced_query])
            scores = (query_vec @ self.tfidf_matrix.T).toarray()[0]
            top_n = np.argsort(scores)[-n_results:][::-1]
            
            filtered_results = []
            for idx in top_n:
                doc = self.documents[idx]
                doc_lower = doc.lower()
                
                matches_all = True
                for field, value in query_parts.items():
                    if value and value not in doc_lower:
                        matches_all = False
                        break
                    
                if matches_all:
                    filtered_results.append(doc)
            
            logger.info(f"Found {len(filtered_results)} course matches for query: {query_text}")
            return {"documents": [filtered_results[:n_results]]}
                
        except Exception as e:
            logger.error(f"Error during course search: {e}")
            return {"documents": [["Error occurred during search"]]}