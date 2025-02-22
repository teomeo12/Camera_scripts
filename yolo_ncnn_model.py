import depthai as dai
import cv2
import time
import numpy as np
import ncnn  # <-- NCNN python bindings
import supervision as sv

# -------------- NCNN YOLO Wrapper --------------
class NCNNYoloDetector:
    def __init__(self, param_path, bin_path, input_size=(640, 640), conf_thresh=0.25, nms_thresh=0.45):
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # Optional: if your net requires specific settings
        self.net.opt.use_vulkan_compute = False  # or True if Vulkan available

    def detect(self, frame_bgr):
        """
        Run inference on a BGR image using the NCNN model.
        Returns bounding boxes, confidence scores, and class IDs.
        Each array is shape (N,) or (N,4).
        """

        # 1) Preprocess
        in_w, in_h = self.input_size
        img_h, img_w = frame_bgr.shape[:2]

        # Letterbox resize to [in_w, in_h]
        scale = min(in_w / img_w, in_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h))
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Convert to NCNN format
        mat = ncnn.Mat.from_pixels(canvas.tobytes(), ncnn.Mat.PixelType.PIXEL_BGR, in_w, in_h)

        # 2) Forward pass
        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        # If you have multiple threads or want to speed up:
        # ex.set_num_threads(4)

        ex.input("images", mat)
        ret, out = ex.extract("output")  # The name "output" depends on your model’s final layer

        # 3) Parse outputs
        # Your model’s final layer might be Nx(7) or Nx(6) with [x1, y1, x2, y2, score, class, ...].
        # This part heavily depends on how your YOLOv10 was exported. 
        # Let's assume Nx6: x1, y1, x2, y2, score, class
        # You’ll have to adjust if your final layer differs.
        
        bboxes = []
        confs = []
        class_ids = []

        for i in range(out.h):
            row = out.row(i)
            x1 = row[0]
            y1 = row[1]
            x2 = row[2]
            y2 = row[3]
            score = row[4]
            cls = row[5]

            if score < self.conf_thresh:
                continue

            # Convert coords back to original image size
            # Undo letterbox
            box_w = (x2 - x1)
            box_h = (y2 - y1)
            # scale coords to the new_w/new_h
            x1 = (x1 * (new_w / in_w))
            y1 = (y1 * (new_h / in_h))
            box_w = (box_w * (new_w / in_w))
            box_h = (box_h * (new_h / in_h))

            # shift coords to match letterbox area
            x1 -= (in_w - new_w) / 2 if new_w < in_w else 0
            y1 -= (in_h - new_h) / 2 if new_h < in_h else 0

            # clip coords in range
            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = x1 + box_w
            y2 = y1 + box_h
            x2 = np.clip(x2, 0, img_w - 1)
            y2 = np.clip(y2, 0, img_h - 1)

            bboxes.append([x1, y1, x2, y2])
            confs.append(score)
            class_ids.append(cls)

        if len(bboxes) == 0:
            return np.empty((0,4)), np.empty(0), np.empty(0)

        # 4) Optional: NMS step. You can either implement your own or rely on supervision’s Detections
        # For brevity, let's skip custom NMS here and rely on supervision if desired
        return np.array(bboxes), np.array(confs), np.array(class_ids, dtype=int)


def main():
    """
    This is your main function, modified to use the NCNNYoloDetector instead of ultralytics.YOLO().
    """
    # Create your NCNN-based YOLO detector
    ncnn_detector = NCNNYoloDetector(
        param_path="/home/teo/Desktop/YoLO10_model_divot_detection/model.ncnn.param",
        bin_path="/home/teo/Desktop/YoLO10_model_divot_detection/model.ncnn.bin",
        input_size=(640, 640),   # adjust if your model uses a different input size
        conf_thresh=0.25,
        nms_thresh=0.45
    )

    # Supervision annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # OAK-D Lite: Setup the RGB camera node
    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Setup mono cameras for depth
    monoLeft = pipeline.createMonoCamera()
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    monoRight = pipeline.createMonoCamera()
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    # Performance parameters
    process_every_n_frames = 2
    frame_counter = 0
    last_detections = None

    # FPS measurement
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

            # Capture frames
            inRgb = qRgb.get()
            inDepth = qDepth.get()
            frameRgb = inRgb.getCvFrame()
            depthFrame = inDepth.getFrame()

            # Ensure depthFrame matches color frame shape if needed
            if depthFrame.shape[:2] != frameRgb.shape[:2]:
                depthFrame = cv2.resize(depthFrame, (frameRgb.shape[1], frameRgb.shape[0]))

            # Only run inference every nth frame
            if frame_counter % process_every_n_frames == 0:
                bboxes, confs, class_ids = ncnn_detector.detect(frameRgb)
                # Convert to Supervision Detections
                if len(bboxes) > 0:
                    xyxy = np.array(bboxes, dtype=float)
                    # Supervision expects shape (N,4)
                    detections = sv.Detections(
                        xyxy=xyxy,
                        confidence=confs,
                        class_id=class_ids
                    )
                else:
                    detections = sv.Detections(
                        xyxy = np.empty((0,4)),
                        confidence = np.empty((0,)),
                        class_id = np.empty((0,))
                    )
                last_detections = detections
            else:
                detections = last_detections

            # If no detections, create empty to avoid errors
            if detections is None:
                detections = sv.Detections(
                    xyxy = np.empty((0,4)),
                    confidence = np.empty((0,)),
                    class_id = np.empty((0,))
                )

            # Annotate depth info
            if detections.xyxy.size > 0:
                for xy in detections.xyxy:
                    xmin, ymin, xmax, ymax = xy[:4]
                    center_x = int((xmin + xmax)/2)
                    center_y = int((ymin + ymax)/2)
                    if 0 <= center_y < depthFrame.shape[0] and 0 <= center_x < depthFrame.shape[1]:
                        depth_value = depthFrame[center_y, center_x]
                        cv2.putText(frameRgb, f"{depth_value} mm", (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Annotate frame with bounding boxes and labels
            annotated_image = box_annotator.annotate(scene=frameRgb.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            # Show FPS
            cv2.putText(annotated_image, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("OAK-D Lite Divot Detection (NCNN)", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
