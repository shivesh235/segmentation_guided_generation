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

def draw_number_label(image, box, number):
    """Draw a number label with a background circle on the image."""
    x1, y1 = int(box[0]), int(box[1])  # Use top-left corner of bounding box
    
    # Convert numpy array to PIL Image for drawing
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    
    # Calculate circle position and size
    circle_radius = 12
    circle_x = x1
    circle_y = y1
    
    # Draw white circle background
    draw.ellipse(
        [
            circle_x - circle_radius, 
            circle_y - circle_radius,
            circle_x + circle_radius, 
            circle_y + circle_radius
        ],
        fill='white',
        outline='black'
    )
    
    # Draw number
    # Use default font since custom fonts might not be available
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = circle_x - text_width // 2
    text_y = circle_y - text_height // 2
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

        # Get bounding boxes, class names, and confidence scores
        boxes = results[0].boxes.data.cpu().numpy()
        class_names = results[0].names
        
        # Prepare detection results and draw numbers
        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, class_id = box
            detections.append({
                "id": i,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class": class_names[int(class_id)]
            })
            # Draw number label on the image
            labeled_image = draw_number_label(labeled_image, box, i + 1)

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

        # Apply the combined mask
        masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

        # Convert to base64
        buffered = BytesIO()
        Image.fromarray(masked_image).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse({
            "segmented_image": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse({"error": str(e)})