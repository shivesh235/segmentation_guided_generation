from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import logging

app = FastAPI()

# Load YOLOv8 Model
model = YOLO('yolov8s-seg.pt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the image from upload
        image = np.array(Image.open(BytesIO(await file.read())))

        # Ensure image is in correct format (RGB)
        if image.shape[-1] == 4:  # if the image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 1:  # grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Run YOLOv8 model for segmentation
        results = model(image)

        # Ensure there are results and masks
        if not results or not results[0].masks:
            return {"error": "No objects detected or no segmentation masks found."}

        # Assuming the user selects the first detected object's mask
        mask = results[0].masks.data[0].cpu().numpy()  # Ensure it's a numpy array

        # Resize the mask to match the original image dimensions
        original_height, original_width = image.shape[:2]
        resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Apply the resized mask to the original image
        masked_image = apply_mask(image, resized_mask)

        # Convert the masked image to a format for display
        masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
        buffer = BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"error": str(e)}


def apply_mask(image, mask):
    """
    Apply the segmentation mask to the image.
    """
    try:
        # Convert mask to binary (1 or 0)
        binary_mask = (mask > 0.5).astype(np.uint8)

        # If the image is not in the expected shape, log and handle it
        if len(image.shape) == 2:  # grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if len(binary_mask.shape) == 3 and binary_mask.shape[-1] == 1:
            binary_mask = binary_mask[:, :, 0]  # Convert (H, W, 1) to (H, W)

        # Apply mask on the image (create a masked version of the image)
        masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

        # Convert to RGB if needed (OpenCV reads images in BGR by default)
        masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        return masked_image_rgb
    
    except Exception as e:
        logger.error(f"Error applying mask: {e}")
        raise

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
