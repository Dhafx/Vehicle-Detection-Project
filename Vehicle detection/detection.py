import cv2
from ultralytics import YOLO
import numpy as np
from sort import *

def load_model():
    return YOLO("../weights/yolov8l.pt")

def load_video(video_path):
    return cv2.VideoCapture(video_path)

def load_mask(mask_path, img_shape):
    mask = cv2.imread(mask_path)
    return cv2.resize(mask, (img_shape[1], img_shape[0]))

def process_frame(frame, mask, model):
    interest_area = cv2.bitwise_and(frame, mask)
    results = model(interest_area, stream=True)
    detections = np.empty((0, 5))

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            cls = classNames[cls_id]

            if cls in ["bus", "truck", "car", "motorbike"] and confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls}, id : {cls_id:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                              (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)

                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                current_array = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_array))

    return frame, detections

def update_tracker(tracker, detections):
    return tracker.update(detections)

def count_vehicles(gate, tracker_output, total_count, frame):
    for r in tracker_output:
        x1, y1, x2, y2, id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if gate[0] < center_x < gate[2] and gate[1] < center_y < gate[3]:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(frame, (gate[0], gate[1]), (gate[2], gate[3]), (0, 255, 0), 6)
    return total_count

def draw_gate(frame, gate):
    cv2.line(frame, (gate[0], gate[1]), (gate[2], gate[3]), (0, 0, 255), 6)

def draw_text(frame, total_count):
    text = f"Vehicle Count: {len(total_count)}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = 10
    text_y = 30

    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
def main():
    model = load_model()
    video_path = "../Media/toll_gate.mp4"
    mask_path = "../Media/mask.png"
    gate = [28, 163, 552, 286]

    vid = load_video(video_path)
    tracker = Sort(max_age=20)
    total_count = []

    # Get video properties
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        success, frame = vid.read()
        if not success:
            print("Failed to read frame from video or end of video reached.")
            break

        if frame is not None and frame.size > 0:
            mask = load_mask(mask_path, frame.shape)
            frame, detections = process_frame(frame, mask, model)
            tracker_output = update_tracker(tracker, detections)
            draw_gate(frame, gate)
            total_count = count_vehicles(gate, tracker_output, total_count, frame)
            draw_text(frame, total_count)

            out.write(frame)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
