from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8s-seg.pt')

# Load an image
image_path = 'download.png'
image = cv2.imread(image_path)

# Perform inference
results = model(image, save=True)

# Visualize results (optional)
results[0].plot()

# Save the mask of the segmented object
masks = results[0].masks  # Masks for detected objects
