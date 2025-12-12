# schemas.py
from pydantic import BaseModel, Field

# --- Request Schema: Data coming IN from the frontend ---
class ImageGenerationRequest(BaseModel):
    """
    Defines the input parameters for the Stable Diffusion generation endpoint.
    All fields use Pydantic validation and provide default values.
    """
    
    # Required Fields
    prompt: str = Field(
        ..., 
        description="The positive text prompt for image generation.",
        min_length=5,
        max_length=500
    )
    
    # Optional Fields
    negative_prompt: str = Field(
        "", 
        description="A prompt to guide the model away from.",
        max_length=500
    )
    
    # Sampler/Inference Settings
    num_inference_steps: int = Field(
        50, 
        description="Number of diffusion steps (higher = better quality, slower).",
        ge=10, 
        le=150
    )
    guidance_scale: float = Field(
        7.5, 
        description="How much the prompt guides the image (higher = more specific).",
        ge=1.0, 
        le=20.0
    )
    
    # Image Configuration
    width: int = Field(
        768, 
        description="Width of the output image (768 for SD 2.1).",
        ge=512, 
        le=1024
    )
    height: int = Field(
        768, 
        description="Height of the output image (768 for SD 2.1).",
        ge=512, 
        le=1024
    )
    
    # Seed
    seed: int = Field(
        -1, 
        description="Random seed for reproducible results (-1 for random).",
        ge=-1
    )

# --- Response Schema: Data going OUT to the frontend ---
class ImageGenerationResponse(BaseModel):
    """
    Defines the output structure for the successful response.
    """
    image_base64: str = Field(
        ..., 
        description="The generated image encoded as a Base64 string."
    )
    endpoint_name: str = Field(
        ...,
        description="The SageMaker endpoint that processed the request."
    )