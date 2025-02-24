import depthai as dai
import cv2
import numpy as np
import time
import supervision as sv
from ultralytics import YOLO

def main():
    # Load your YOLO model (update the path as needed)
    #model_path = "/home/teo/Desktop/YoLO10_model_divot_detection/best.pt"
    model_path = r'C:\Users\teomeo\Desktop\aMU_MSc\desertation\YoLO10_model_divot_detection\best.pt'

    #model_path = "/home/teo/Desktop/YoLO10_model_divot_detection/best_ncnn_model/model_ncnn.py"
    model = YOLO(model_path)

    # Initialize annotators (using BoxAnnotator and LabelAnnotator)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # Setup the RGB camera node
    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Setup mono cameras for depth computation
    monoLeft = pipeline.createMonoCamera()
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight = pipeline.createMonoCamera()
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # Create the stereo depth node (depth output is auto-enabled)
    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    # Frame skipping and FPS measurement parameters
    process_every_n_frames = 2  # process every 2nd frame
    frame_counter = 0
    last_detections = None

    fps_timer = time.time()
    fps_count = 0
    current_fps = 0

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            frame_counter += 1
            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                current_fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            # Get frames from the device
            inRgb = qRgb.get()
            inDepth = qDepth.get()

            frameRgb = inRgb.getCvFrame()
            depthFrame = inDepth.getFrame()  # depth values in millimeters

            # Resize depth frame to match RGB frame if needed
            if depthFrame.shape[:2] != frameRgb.shape[:2]:
                depthFrame = cv2.resize(depthFrame, (frameRgb.shape[1], frameRgb.shape[0]))

            # Run YOLO inference only every nth frame to save processing time
            if frame_counter % process_every_n_frames == 0:
                results = model(frameRgb)
                last_detections = sv.Detections.from_ultralytics(results[0])
                detections = last_detections
            else:
                detections = last_detections

            # If no detections, replace with empty detections to avoid errors
            if detections is None:
                detections = sv.Detections(
                    xyxy = np.empty((0,4)),
                    confidence = np.empty((0,)),
                    class_id = np.empty((0,))
                )

            # For each detection, calculate center and overlay depth value
            if detections.xyxy.size > 0:
                for bbox in detections.xyxy:
                    # bbox: [xmin, ymin, xmax, ymax, ...]
                    xmin, ymin, xmax, ymax = bbox[:4]
                    center_x = int((float(xmin) + float(xmax)) / 2)
                    center_y = int((float(ymin) + float(ymax)) / 2)
                    if 0 <= center_y < depthFrame.shape[0] and 0 <= center_x < depthFrame.shape[1]:
                        depth_value = depthFrame[center_y, center_x]
                        cv2.putText(frameRgb, f"{depth_value} mm", (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Annotate the frame with boxes and labels
            annotated_image = box_annotator.annotate(scene=frameRgb.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            # Display current FPS on the frame
            cv2.putText(annotated_image, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the annotated image
            cv2.imshow("OAK-D Lite Divot Detection", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
