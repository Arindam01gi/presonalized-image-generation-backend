import os
import json
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from PIL import Image
from controlnet_aux import OpenposeDetector
from transformers import CLIPVisionModelWithProjection

# --- 1. model_fn (Loads the complex, large model) ---
def model_fn(model_dir):
    """
    Loads the combined Stable Diffusion (SD 1.5), ControlNet (OpenPose), and IP-Adapter pipeline.
    This step is the cause of the long cold start due to the size (~5.5 GB).
    """
    try:
        # Define paths based on the structure created in model.tar.gz
        sd_path = os.path.join(model_dir, "stable-diffusion-v1-5")
        controlnet_path = os.path.join(model_dir, "controlnet")
        ip_adapter_dir = os.path.join(model_dir, "ip_adapter_models")
        
        # 1. Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16
        )
        
        # 2. Load Base Pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, 
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        
        # 3. Load IP-Adapter
        image_encoder_path = os.path.join(ip_adapter_dir, "image_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(pipe.device)
        pipe.load_ip_adapter(
            ip_adapter_dir, 
            subfolder="", 
            weight_name="ip-adapter-plus-face_sd15.bin", 
            image_encoder=image_encoder
        )
        
        # Run on CPU since Serverless does not support GPU
        pipe.to("cpu")
        pipe.enable_xformers_memory_efficient_attention() # Optimization is key!
        
        return pipe

    except Exception as e:
        print(f"Model Loading Error: {e}")
        raise e

# --- 2. input_fn (Processes the request from the FastAPI Lambda) ---
def input_fn(request_body, content_type):
    """Parses the incoming JSON request."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported Content Type: {content_type}")

# --- 3. predict_fn (Generates the image) ---
def predict_fn(data, pipe):
    """Runs the full SD + ControlNet + IP-Adapter inference."""
    
    # Pre-process the input images from Base64
    pose_img_bytes = base64.b64decode(data['pose_image'])
    face_ref_bytes = base64.b64decode(data['face_reference'])
    pose_image = Image.open(BytesIO(pose_img_bytes)).convert("RGB")
    face_reference = Image.open(BytesIO(face_ref_bytes)).convert("RGB")

    # OpenPose preprocessing (must be done inside the container or pre-processed by client)
    # Instantiate the detector inside predict_fn to prevent memory issues if model_fn is too crowded,
    # or ideally, move this initialization to model_fn.
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet") 
    control_image = openpose(pose_image)

    # Run the combined pipeline
    generated_image = pipe(
        prompt=data['prompt'],
        negative_prompt=data.get('negative_prompt', ""),
        image=control_image,
        ip_adapter_image=face_reference,
        num_inference_steps=data.get('steps', 20),
        guidance_scale=data.get('guidance_scale', 7.5)
    ).images[0]

    # Encode the output image back to base64
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"generated_image_base64": img_str}

# --- 4. output_fn (Formats the final response) ---
def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")