import os
import cv2
import yaml
from inference import Lanefinder
from glob import glob
import time
import numpy as np

def read_config():
    if not os.path.isfile('config.yaml'):
        raise FileNotFoundError('Could not find config file')

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config
def main():
    config = read_config()

    # Lấy đường dẫn ảnh duy nhất từ config
    input_image = config.get("input_image", None)
    if input_image is None or not os.path.isfile(input_image):
        raise FileNotFoundError(f"Could not find input image: {input_image}")

    output_folder = config.get("output_image", "results")
    os.makedirs(output_folder, exist_ok=True)

    # Khởi tạo lane detector
    lanefinder = Lanefinder(
        model_path=config['model_path'],
        input_shape=config['input_shape'],
        output_shape=tuple(config['output_shape']),
        quant=config['quantization'],
        dequant=config['dequantization']
    )

    # Đọc và xử lý đúng một ảnh
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"Failed to read image: {input_image}")

    frmcpy = img.copy()
    pil_img = lanefinder._preprocess(img)

    if lanefinder._interpreter is not None:
        from pycoral.adapters.common import set_input

        set_input(lanefinder._interpreter, pil_img)
        start = time.perf_counter()
        lanefinder._interpreter.invoke()
        duration_ms = (time.perf_counter() - start) * 1000
        print(f'Inference Time: {duration_ms:.2f} ms')
        output = lanefinder._interpreter.get_tensor(
            lanefinder._interpreter.get_output_details()[0]['index']
        )
        camera_offset_bias = 0
        result = lanefinder._postprocess(output, frmcpy, camera_offset_bias)
    else:
        # Nếu không có TPU
        result = cv2.putText(
            cv2.resize(frmcpy, lanefinder._output_shape),
            'TPU not detected!',
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2
        )

    # Lưu ảnh kết quả
    filename = os.path.basename(input_image)

    base,ext = os.path.splitext(filename)
    new_name = f"{base}_proces{ext}"
    out_path = os.path.join(output_folder, new_name)
    cv2.imwrite(out_path, result)
    # out_path = os.path.join(output_folder, filename)
    # cv2.imwrite(out_path, result)
    print(f"[INFO] Saved result to: {out_path}")

    lanefinder.destroy()
    print("[INFO] Done.")

if __name__ == '__main__':
    main()
