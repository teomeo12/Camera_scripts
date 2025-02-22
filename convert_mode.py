from ultralytics import YOLO

# Load your YOLO model from the specified location
model = YOLO("/home/teo/Desktop/YoLO10_model_divot_detection/best.pt")

# Export the model to NCNN format
model.export(format="ncnn")
