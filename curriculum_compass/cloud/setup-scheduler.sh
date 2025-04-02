#!/bin/bash

# Set your GCP project ID
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"

# Create a service account for the Cloud Scheduler
echo "Creating service account for Cloud Scheduler..."
gcloud iam service-accounts create curriculum-compass-scheduler \
    --display-name="Curriculum Compass Scheduler Service Account"

# Grant the service account permission to invoke Cloud Functions
echo "Granting function invoker role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:curriculum-compass-scheduler@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudfunctions.invoker"

# Get the URL of the course scraper function
FUNCTION_URL=$(gcloud functions describe scrape-courses-function \
    --gen2 \
    --region=$REGION \
    --format="value(url)")

# Create a Cloud Scheduler job for weekly course updates (Sunday at midnight)
echo "Creating Cloud Scheduler job for weekly course updates..."
gcloud scheduler jobs create http update-course-data-monthly \
    --schedule="0 0 1 * *" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --oidc-service-account="curriculum-compass-scheduler@$PROJECT_ID.iam.gserviceaccount.com" \
    --oidc-token-audience="$FUNCTION_URL" \
    --message-body='{"subject":"CS"}'

echo "Cloud Scheduler setup complete. Weekly data scraping will run every Sunday at midnight."