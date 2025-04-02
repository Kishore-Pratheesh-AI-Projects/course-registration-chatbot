#!/bin/bash

# Set your GCP project ID
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"

# 1. Deploy Course Scraper Function
echo "Deploying Course Scraper Function..."
gcloud functions deploy scrape-courses-function \
    --gen2 \
    --runtime=python310 \
    --region=$REGION \
    --source=./functions/scraper/ \
    --entry-point=scrape_courses_function \
    --trigger-http \
    --timeout=540s \
    --memory=2048MB \
    --set-env-vars="BUCKET_NAME=curriculum-compass-raw-data"

# 2. Deploy Data Processor Function
echo "Deploying Data Processor Function..."
gcloud functions deploy process-data-function \
    --gen2 \
    --runtime=python310 \
    --region=$REGION \
    --source=./functions/processor/ \
    --entry-point=process_data_function \
    --trigger-event=google.cloud.storage.object.v1.finalized \
    --trigger-resource=curriculum-compass-raw-data \
    --timeout=300s \
    --memory=2048MB \
    --set-env-vars="OUTPUT_BUCKET=curriculum-compass-processed-data"

# 3. Deploy Embedding Generator Function
echo "Deploying Embedding Generator Function..."
gcloud functions deploy generate-embeddings-function \
    --gen2 \
    --runtime=python310 \
    --region=$REGION \
    --source=./functions/embedding/ \
    --entry-point=generate_embeddings_function \
    --trigger-event=google.cloud.storage.object.v1.finalized \
    --trigger-resource=curriculum-compass-processed-data \
    --timeout=540s \
    --memory=4096MB \
    --set-env-vars="PROJECT_ID=$PROJECT_ID,REGION=$REGION,EMBEDDING_MODEL=all-MiniLM-L6-v2,VECTOR_INDEX_ID=curriculum-compass-course-index,EMBEDDINGS_BUCKET=curriculum-compass-embeddings"

echo "All Cloud Functions deployed successfully!"