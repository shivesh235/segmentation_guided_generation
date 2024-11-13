from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the inpainting pipeline for CPU
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
)
pipeline.to("cpu")  # Set the pipeline to use CPU

class SelectionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    x: int  # Starting x coordinate
    y: int  # Starting y coordinate
    width: int
    height: int
    prompt: str

def base64_to_image(base64_str: str):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str.split(",")[1])
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_mask(width: int, height: int, x: int, y: int, box_width: int, box_height: int) -> Image.Image:
    """Create a mask image with the selected region"""
    mask = Image.new('RGB', (width, height), 'black')
    mask_draw = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    cv2.rectangle(mask_draw, (x, y), (x + box_width, y + box_height), (255, 255, 255), -1)
    return Image.fromarray(cv2.cvtColor(mask_draw, cv2.COLOR_BGR2RGB))

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Endpoint to upload and return the original image"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return JSONResponse({
        "image": image_to_base64(image),
        "width": image.width,
        "height": image.height
    })

@app.post("/inpaint/")
async def inpaint_region(request: SelectionRequest):
    """Endpoint to perform inpainting on selected region"""
    # Convert base64 to image
    init_image = base64_to_image(request.image_data)
    
    # Create mask for selected region
    mask_image = create_mask(
        init_image.width,
        init_image.height,
        request.x,
        request.y,
        request.width,
        request.height
    )
    
    # Ensure images are the correct size (multiple of 8)
    def process_image(img: Image.Image) -> Image.Image:
        w, h = img.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        return img.resize((w, h))

    init_image = process_image(init_image)
    mask_image = process_image(mask_image)
    
    # Generate image on CPU
    output_image = pipeline(
        prompt=request.prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=50
    ).images[0]
    
    return JSONResponse({
        "original_image": image_to_base64(init_image),
        "mask_image": image_to_base64(mask_image),
        "result_image": image_to_base64(output_image)
    })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
