import depthai as dai
import cv2
import sys

# Add your YOLO model directory to the Python path
sys.path.append('/home/teo/Desktop/YoLO10_model_divot_detection')
from divot_detection import run_inference  # Adjust the function name as defined in your script

# Create a DepthAI pipeline
pipeline = dai.Pipeline()

# Create a ColorCamera node
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)

# Create an XLinkOut node to stream RGB data
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    while True:
        inRgb = qRgb.get()  # Wait for an RGB frame
        frame = inRgb.getCvFrame()
        
        # Run YOLO inference on the frame using your custom function
        annotated_frame = run_inference(frame)
        
        cv2.imshow("Divot Detection", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
