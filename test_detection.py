import os
import time
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# --- Configuration ---
MODEL_PATH = "best_full_integer_quant_edgetpu.tflite"
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "results"

CLASS_NAMES = {
    0: ("Car", 2.0, 7.5),
    1: ("Bus", 2.5, 7.5),
    2: ("Truck", 2.5, 7.5),
    3: ("Motorcycle", 0.8, 7.5),
    4: ("Person", 0.5, 7.5),
    5: ("Bicycle", 0.8, 7.5),
}

NUM_CLASSES = 6
ALERT_COLOR = (0, 0, 255)
WARNING_COLOR = (0, 255, 255)
SAFE_COLOR = (0, 255, 0)
ALERT_ALPHA = 0.15
FOCAL_LENGTH = 350.0
CONFIDENCE_THRESHOLD = 0.55
NMS_IOU_THRESHOLD = 0.4
NMS_MAX_BOXES = 20
PROC_WIDTH = 640
PROC_HEIGHT = 360

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load TFLite model and allocate tensors
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_size = common.input_size(interpreter)
print(f"[INFO] Model input size: {input_size}")

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

model_input_height = input_details['shape'][1]
model_input_width = input_details['shape'][2]
input_index = input_details['index']
input_dtype = input_details['dtype']
output_index = output_details['index']

def preprocess_image_tflite_pad(image, target_h, target_w, dtype):
    """
    Resize and pad an image to (target_h, target_w), then normalize or convert dtype.
    Returns:
      - input_tensor suitable for TFLite interpreter
      - scale factor (used to reverse scaling on detections)
      - offset dw, dh (padding amounts)
    """
    h, w, _ = image.shape
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(image, (new_w, new_h))

    padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded_img[dh:dh + new_h, dw:dw + new_w, :] = resized_img

    if dtype == np.float32:
        input_tensor = padded_img.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        input_tensor = padded_img.astype(np.uint8)
    else:
        input_tensor = padded_img.astype(dtype)

    return np.expand_dims(input_tensor, axis=0), scale, dw, dh

def estimate_distance(real_width, pixel_width, focal_length):
    """
    Simple pinhole camera formula to estimate distance (in same units as real_width and focal_length).
    Avoid division by zero by returning infinity if pixel_width <= 0.
    """
    return (real_width * focal_length) / pixel_width if pixel_width > 0 else float('inf')

def decode_yolo_boxes_normalized_input(boxes):
    """
    Convert YOLO-style [x_center, y_center, width, height] (normalized) to [ymin, xmin, ymax, xmax].
    boxes: shape (N,4)
    """
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ymin = y_c - h / 2
    xmin = x_c - w / 2
    ymax = y_c + h / 2
    xmax = x_c + w / 2
    return np.stack([ymin, xmin, ymax, xmax], axis=1)

def non_max_suppression(boxes, scores, classes, iou_threshold=0.4, max_boxes=20):
    """
    Basic NMS implementation: 
    - boxes: array of [ymin, xmin, ymax, xmax]
    - scores: corresponding confidence scores
    - returns indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0 and len(keep) < max_boxes:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-10)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)

# Iterate through all image files in INPUT_FOLDER
for filename in os.listdir(INPUT_FOLDER):
    # Only process common image extensions
    if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png") or filename.lower().endswith(".bmp")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Read the image
    frame_orig = cv2.imread(input_path)
    if frame_orig is None:
        print(f"Warning: could not read image {input_path}. Skipping.")
        continue

    # Resize the frame to a fixed processing size
    frame = cv2.resize(frame_orig, (PROC_WIDTH, PROC_HEIGHT))

    # Preprocess for TFLite model
    input_tensor, scale, dw, dh = preprocess_image_tflite_pad(frame, model_input_height, model_input_width, input_dtype)

    # Run inference
    interpreter.set_tensor(input_index, input_tensor)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    # print(f"Inference time for {filename}: {inference_time:.3f} seconds")

    output_data = interpreter.get_tensor(output_index)


    # Decode raw TFLite output
    boxes_norm = []
    scores = []
    class_ids = []

    # Some TFLite models return shape (1, N, 4+NUM_CLASSES); handle that
    if output_data.shape[0] == 1:
        out = np.squeeze(output_data)
        # If the shape is (N, > 4+NUM_CLASSES), maybe it's already transposed; check and transpose
        if out.shape[1] > NUM_CLASSES + 4:
            out = out.T
        raw_boxes = out[:, :4]               # normalized [x_center, y_center, w, h]
        raw_scores = out[:, 4:]              # logits for each class
        probs = 1 / (1 + np.exp(-raw_scores)) # apply sigmoid to get probabilities
        scores_all = np.max(probs, axis=1)
        class_ids_all = np.argmax(probs, axis=1)
        valid_inds = np.where(scores_all >= CONFIDENCE_THRESHOLD)[0]
        if len(valid_inds) > 0:
            boxes_norm = raw_boxes[valid_inds]
            scores = scores_all[valid_inds]
            class_ids = class_ids_all[valid_inds]

    # Run NMS if we have any candidates
    if len(boxes_norm) > 0:
        boxes_decoded = decode_yolo_boxes_normalized_input(np.array(boxes_norm))
        keep_inds = non_max_suppression(boxes_decoded, np.array(scores), np.array(class_ids), NMS_IOU_THRESHOLD, NMS_MAX_BOXES)
        final_boxes = boxes_decoded[keep_inds]
        final_scores = np.array(scores)[keep_inds]
        final_classes = np.array(class_ids)[keep_inds]
        if np.any(final_classes[:5] == 1):
            print('founded')
            indices = np.where(final_classes == 1)[0]
            print(final_scores)
            print(final_boxes)
            for idx in indices:
                # ensure idx is a Python int (not an array-slice)
                idx = int(idx)
                class_name, real_w, safe_d = CLASS_NAMES[idx]
                print(f"{filename}: found a “{class_name}” at index {idx}")
            break
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_classes = np.array([])

    vehicle_close_flag = False

    # Annotate detections back onto the resized frame
    for i in range(len(final_boxes)):
        y_min, x_min, y_max, x_max = final_boxes[i]
        # Convert normalized coords + padding back to pixel coords on the PROC_WIDTH x PROC_HEIGHT frame
        x1 = int((x_min * model_input_width - dw) / scale)
        y1 = int((y_min * model_input_height - dh) / scale)
        x2 = int((x_max * model_input_width - dw) / scale)
        y2 = int((y_max * model_input_height - dh) / scale)

        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(PROC_WIDTH - 1, x2), min(PROC_HEIGHT - 1, y2)
        pixel_width = x2 - x1
        if pixel_width <= 0:
            continue

        class_id = int(final_classes[i])
        if class_id not in CLASS_NAMES:
            continue

        class_name, real_width, safe_dist = CLASS_NAMES[class_id]
        distance = estimate_distance(real_width, pixel_width, FOCAL_LENGTH)

        # Determine color and overlay for distance thresholding
        if distance < safe_dist:
            color = ALERT_COLOR
            vehicle_close_flag = True
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), ALERT_COLOR, -1)
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], ALERT_ALPHA, frame[y1:y2, x1:x2], 1 - ALERT_ALPHA, 0)
        elif distance < safe_dist * 2:
            color = WARNING_COLOR
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), WARNING_COLOR, -1)
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], ALERT_ALPHA, frame[y1:y2, x1:x2], 1 - ALERT_ALPHA, 0)
        else:
            color = SAFE_COLOR

        # Draw bounding box and label
        label = f"{class_name} - {distance:.1f}m"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x1 + max(0, (pixel_width - text_width) // 2)
        text_y = y1 - 10 if (y1 - 10) > text_height else (y1 + text_height + 5)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # If any vehicle is too close, overlay a "Vehicle close!" warning
    if vehicle_close_flag:
        cv2.putText(frame, "Vehicle close!", (10, PROC_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ALERT_COLOR, 2)

    # Save the annotated result
    cv2.imwrite(output_path, frame)
    # print(f"Processed {filename} -> {output_path}")
