import io
import os
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "your-default-s3-bucket")
SADEMAKER_ENDPOINT = os.environ.get("SADEMAKER_ENDPOINT", "your-default-sagemaker-endpoint")

@app.get("/")
async def root():
    return {"message": "FastAPI Proxy Running. Access /personalize for image processing."}

@app.post("/personalize")
async def personalize_image(file:UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()
    print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes. Simulating SageMaker call...")

    mock_s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/mock/result_{file.filename}.jpg"

    return JSONResponse(content={"illustration_url": mock_s3_url})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)