import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Create a ColorCamera node
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)

# Create an XLinkOut node to stream data
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    while True:
        inRgb = qRgb.get()  # Blocking call, waits for new data
        frame = inRgb.getCvFrame()
        cv2.imshow("OAK-D Lite Preview", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
