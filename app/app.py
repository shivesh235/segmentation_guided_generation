from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List
from ultralytics import YOLO
import base64
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 Model
model = YOLO('yolov8s-seg.pt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationRequest(BaseModel):
    selected_indices: List[int]
    image_data: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

def draw_number_label(image, mask, number):
    """Draw a number label at the centroid of the mask."""
    # Calculate mask centroid
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        # Fallback to mask bounds if moments calculation fails
        y, x = np.where(mask)
        if len(x) > 0 and len(y) > 0:
            centroid_x = int(np.mean(x))
            centroid_y = int(np.mean(y))
        else:
            return image

    # Convert numpy array to PIL Image for drawing
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    
    # Calculate circle position and size
    circle_radius = 12
    
    # Draw white circle background
    draw.ellipse(
        [
            centroid_x - circle_radius, 
            centroid_y - circle_radius,
            centroid_x + circle_radius, 
            centroid_y + circle_radius
        ],
        fill='white',
        outline='black'
    )
    
    # Draw number
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = centroid_x - text_width // 2
    text_y = centroid_y - text_height // 2
    draw.text((text_x, text_y), text, fill='black')
    
    return np.array(img_pil)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_data = await file.read()
        image = np.array(Image.open(BytesIO(image_data)))
        
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Make a copy of the image for drawing labels
        labeled_image = image.copy()

        # Run YOLOv8 detection
        results = model(image)
        
        if not results or not results[0].masks:
            return JSONResponse({"error": "No objects detected"})

        # Get masks and class information
        masks = results[0].masks
        boxes = results[0].boxes
        class_names = results[0].names
        
        # Prepare detection results and draw numbers
        detections = []
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            mask_array = mask.data[0].cpu().numpy()
            mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]))
            mask_array = mask_array > 0.5

            box_data = box.data.cpu().numpy()[0]
            conf, class_id = box_data[4], int(box_data[5])
            
            detections.append({
                "id": i,
                "confidence": float(conf),
                "class": class_names[class_id]
            })
            
            # Draw number label on the image using mask centroid
            labeled_image = draw_number_label(labeled_image, mask_array, i + 1)

        # Convert labeled image to base64
        buffered = BytesIO()
        Image.fromarray(labeled_image).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "detections": detections,
            "image": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse({"error": str(e)})

@app.post("/segment/")
async def segment_objects(request: SegmentationRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data.split(',')[1])
        image = np.array(Image.open(BytesIO(image_data)))
        
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Run YOLOv8 detection
        results = model(image)
        
        if not results or not results[0].masks:
            return JSONResponse({"error": "No objects detected"})

        # Create combined mask from selected objects
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for idx in request.selected_indices:
            if idx < len(results[0].masks):
                mask = results[0].masks[idx].data[0].cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                combined_mask = cv2.bitwise_or(combined_mask, (mask > 0.5).astype(np.uint8))

        # Create a red overlay
        overlay = image.copy()
        overlay[combined_mask == 1] = [255, 200, 200]  # Light red color

        # Blend the overlay with the original image
        output = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

        # Convert to base64
        buffered = BytesIO()
        Image.fromarray(output).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "segmented_image": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse({"error": str(e)})