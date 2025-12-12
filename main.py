from fastapi import FastAPI
from mangum import Mangum
from config import settings


# FastAPI Application Initialization
app = FastAPI(
    title="Stable Diffusion API",
    version="1.0.0",
    description="Serverless API for Stable Diffusion 2.1 inference via SageMaker Endpoint."
)


# API endpoint
@app.get("/health")
def health_check():
    """Simple endpoint to verify the Lambda function is running."""
    return {
        "status": "ok", 
        "environment": settings.ENVIRONMENT,
        "endpoint": settings.SAGEMAKER_ENDPOINT_NAME
    }

# IMPORTANT: The Mangum adapter handles the translation between 
# AWS Lambda/API Gateway events and the FastAPI application logic.
# This 'handler' variable is what AWS Lambda calls.
handler = Mangum(app)