import io
import os
import boto3
import json
import base64
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT")
REGION_NAME = os.environ.get("AWS_REGION", "ap-south-1")


# Boto3 Client Initialization (Lazy/Global initialization is fine in Lambda)
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)
s3_client = boto3.client("s3", region_name=REGION_NAME)


# --- UTILITY FUNCTIONS ---
def get_presigned_url(key: str) -> str:
    """Generates a temporary, secure URL for the frontend to access the image."""
    return s3_client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": key},
        ExpiresIn=3600,  # URL expires in 1 hour
    )


def upload_to_s3(image_data: bytes, filename: str) -> str:
    """Uploads the generated image bytes to S3."""
    s3_key = f"illustrations/{os.path.basename(filename)}"

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME, Key=s3_key, Body=image_data, ContentType="image/jpeg"
    )

    return s3_key


def invoke_sagemaker_endpoint(input_image_bytes: bytes) -> bytes:
    """Calls the InstantID SageMaker endpoint for image generation."""
    if not SAGEMAKER_ENDPOINT:
        raise Exception("SageMaker Endpoint Name not configured.")
    input_b64 = base64.b64encode(input_image_bytes).decode("utf-8")
    payload = {
        "prompt": "A whimsical, cartoon illustration of a child, high quality, digital art, storybook style",
        "image": input_b64,
        "style_prompt": "fantasy art, detailed, colorful",
        # You would also need a 'controlnet_image' if the template illustration
        # is used as a ControlNet guide. For simplicity, we use the input image
        # only for identity transfer here.
        "seed": 42,
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)
    
    # Assume the model returns a list of base64 encoded images
    generated_image_b64 = result['generated_images'][0]
    generated_image_bytes = base64.b64decode(generated_image_b64)
    
    return generated_image_bytes


@app.get("/")
async def root():
    return {
        "message": "FastAPI Proxy Running. Access /personalize for image processing."
    }


@app.post("/personalize")
async def personalize_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    if not S3_BUCKET_NAME or not SAGEMAKER_ENDPOINT:
        raise HTTPException(status_code=500, detail="Server configuration missing.")
    
    # 1. Read the image file content
    input_image_bytes = await file.read()

    try:
        # 2. Invoke SageMaker Endpoint (InstantID generation)
        generated_image_bytes = invoke_sagemaker_endpoint(input_image_bytes)
        
        # Analysis: This step blocks the Lambda function until the GPU inference 
        # is complete (15-30 seconds). The Lambda maximum timeout (15 minutes) 
        # must be configured to handle this delay.
        
        # 3. Upload result to S3
        s3_key = upload_to_s3(generated_image_bytes, file.filename)
        
        # 4. Generate access URL
        illustration_url = get_presigned_url(s3_key)
        
        return JSONResponse(content={"illustration_url": illustration_url})
        
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
