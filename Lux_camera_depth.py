import depthai as dai
import cv2
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Set up mono (grayscale) cameras for depth
monoLeft = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a StereoDepth node to compute depth
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setOutputDepth(True)
stereo.setOutputRectified(False)  # change to True if you need rectified images
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Create XLinkOut node for depth output
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# Set up the RGB camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Create XLinkOut node for RGB output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()      # Get RGB frame
        inDepth = qDepth.get()  # Get depth frame

        frameRgb = inRgb.getCvFrame()
        # Get depth frame as a numpy array in millimeters
        frameDepth = inDepth.getFrame()

        # Normalize depth for display (optional)
        frameDepthNorm = cv2.normalize(frameDepth, None, 0, 255, cv2.NORM_MINMAX)
        frameDepthNorm = np.uint8(frameDepthNorm)
        frameDepthColor = cv2.applyColorMap(frameDepthNorm, cv2.COLORMAP_JET)

        cv2.imshow("RGB", frameRgb)
        cv2.imshow("Depth", frameDepthColor)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
