# server_mjpeg.py
from flask import Flask, Response
import cv2
import time
import yaml
from inference import Lanefinder
from pycoral.adapters.common import set_input

app = Flask(__name__)

def read_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

@app.route('/mjpeg')
def mjpeg_stream():
    cfg = read_config()
    cap = cv2.VideoCapture(0)
    lanefinder = Lanefinder(
        model_path=cfg['model_path'],
        input_shape=cfg['input_shape'],
        output_shape=tuple(cfg['output_shape']),
        quant=cfg['quantization'],
        dequant=cfg['dequantization'],
    )

    def generate():
        boundary = '--frame'
        fps = 30.0
        frame_duration = 1.0 / fps
        next_ts = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # run inference if available
            if lanefinder._interpreter:
                img_in = lanefinder._preprocess(frame)
                set_input(lanefinder._interpreter, img_in)
                lanefinder._interpreter.invoke()
                out = lanefinder._interpreter.get_tensor(
                    lanefinder._interpreter.get_output_details()[0]['index']
                )
                frame = lanefinder._postprocess(out, frame)

            # JPEG encode
            success, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                continue

            # multipart response chunk
            yield (f"{boundary}\r\n"
                   "Content-Type: image/jpeg\r\n\r\n").encode('utf-8') \
                  + jpeg.tobytes() + b"\r\n"

            # maintain ~30 FPS
            next_ts += frame_duration
            delay = next_ts - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_ts = time.perf_counter()

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host 0.0.0.0 so it’s reachable on your LAN
    app.run(host='0.0.0.0', port=8765, threaded=True)
