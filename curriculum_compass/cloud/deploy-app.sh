#!/bin/bash

# Set your GCP project ID
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"

# Build the container image
echo "Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/curriculum-compass-app ./app

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy curriculum-compass-app \
  --image gcr.io/$PROJECT_ID/curriculum-compass-app \
  --platform managed \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION" \
  --set-env-vars="COURSE_INDEX_ENDPOINT=projects/$PROJECT_ID/locations/$REGION/indexEndpoints/curriculum-compass-course-index-endpoint" \
  --set-env-vars="COURSE_DEPLOYED_INDEX=curriculum-compass-course-index" \
  --set-env-vars="LOGS_BUCKET=curriculum-compass-logs" \
  --allow-unauthenticated

echo "Deployment complete! Your application is now available at:"
gcloud run services describe curriculum-compass-app --region $REGION --format="value(status.url)"