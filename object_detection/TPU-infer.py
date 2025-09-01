import cv2
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
# from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
import time


# Đường dẫn đến model và file labels
MODEL_PATH = 'ssd_mobilenet_v2_coco_quant_postprocess.tflite'
# MODEL_PATH = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
# MODEL_PATH = 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
LABELS_PATH = 'coco_labels.txt'

# Các lớp phương tiện cần giữ lại
VEHICLE_CLASSES = {'car','bus','truck','motorcycle','traffic light','person','bicycle'}

def load_labels(path):
    """
    Đọc file labels dạng:
      0 person
      1 bicycle
      2 car
      ...
    và trả về dict id→tên (vd. 2→"car").
    """
    label_map = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, name = parts
                label_map[int(idx)] = name
    return label_map

def draw_detections(image_cv, detections, labels):
    for obj in detections:
        label = labels[obj.id]
        if label in VEHICLE_CLASSES:
            x0, y0 = int(obj.bbox.xmin), int(obj.bbox.ymin)
            x1, y1 = int(obj.bbox.xmax), int(obj.bbox.ymax)
            cv2.rectangle(image_cv, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(image_cv,
                        f'{label}: {obj.score:.2f}',
                        (x0, y0-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0), 2)


def run_inference(image_path, output_path='output.jpg', score_thresh=0.5):
    # 1. Khởi tạo interpreter và allocate tensors
    # interpreter = make_interpreter(MODEL_PATH)
    interpreter = Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    # 2. Đọc ảnh gốc (PIL) và chuyển sang OpenCV để vẽ
    image_pil = Image.open(image_path).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 3. Resize input và lấy scale để map bounding box
    #    - size: kích thước gốc image_pil.size
    #    - resize: hàm gọi image_pil.resize
    _, scale = common.set_resized_input(
        interpreter,
        image_pil.size,
        lambda new_size: image_pil.resize(new_size, Image.NEAREST)
    )

    # 4. Chạy inference
    t0 = time.perf_counter()
    interpreter.invoke()
    duration_ms = (time.perf_counter() - t0) * 1000
    print(f"[INFO] Inference time: {duration_ms:.2f} ms")

    # 5. Lấy kết quả
    # Thay vì: detections = detect.get_objects(..., score_threshold=score_thresh)
    detections = detect.get_objects(interpreter,
                                    score_threshold=score_thresh,
                                    image_scale=scale)


    # Load labels
    labels = load_labels(LABELS_PATH)

    # **In ra các class đã nhận diện**
    if detections:
        print("Các object đã phát hiện (trên ngưỡng {:.2f}):".format(score_thresh))
        for obj in detections:
            lbl = labels[obj.id]
            print(f" - {lbl:<12} | score = {obj.score:.2f} | bbox = ({obj.bbox.xmin:.2f}, {obj.bbox.ymin:.2f}, {obj.bbox.xmax:.2f}, {obj.bbox.ymax:.2f})")
    else:
        print("Không phát hiện được object nào trên ngưỡng {:.2f}".format(score_thresh))

    # 6. Vẽ khung và lưu ảnh như trước
    draw_detections(image_cv, detections, labels)
    cv2.imwrite(output_path, image_cv)
    print(f'Inference xong, lưu ảnh có detections tại: {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Edge TPU vehicle-only detection')
    parser.add_argument('image', help='Đường dẫn đến ảnh của bạn')
    parser.add_argument('--out', default='output.jpg',
                        help='Đường dẫn lưu ảnh kết quả')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ngưỡng score để lọc detections')
    args = parser.parse_args()

    run_inference(args.image, args.out, args.threshold)