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

    input_folder = config.get("input_folder", "images")
    output_folder = config.get("output_folder", "results")
    
    # input_folder = config.get("small_input", "one_image")
    # output_folder = config.get("small_result", "one_result")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize lane detector
    lanefinder = Lanefinder(
        model_path=config['model_path'],
        input_shape=config['input_shape'],
        output_shape=tuple(config['output_shape']),
        quant=config['quantization'],
        dequant=config['dequantization']
    )

    # Run lane detection on all images
    image_paths = sorted(glob(os.path.join(input_folder, "*.*")))
    print(f"[INFO] Found {len(image_paths)} images in '{input_folder}'")

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Could not read {image_path}")
            continue
        
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
            result = lanefinder._postprocess(output, frmcpy ,camera_offset_bias)
        else:
            result = cv2.putText(
                cv2.resize(frmcpy, lanefinder._output_shape),
                'TPU not detected!',
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2
            )

        # Save output
        filename = os.path.basename(image_path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, result)
        print(f"[INFO] Saved: {out_path}")

    lanefinder.destroy()
    print("[INFO] All done.")


if __name__ == '__main__':
    main()
