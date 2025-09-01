import cv2
cap = cv2.VideoCapture(0)
import asyncio
import time
import numpy as np
import yaml
import websockets
from inference import Lanefinder
import base64
from pycoral.adapters.common import set_input



def read_config():
    with open('config.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


async def video_stream(websocket, path=None):
    config = read_config()

    lanefinder = Lanefinder(
        model_path=config['model_path'],
        input_shape=config['input_shape'],
        output_shape=tuple(config['output_shape']),
        quant=config['quantization'],
        dequant=config['dequantization']
    )
    
    FRAME_INTERVAL = 1.0 / 30.0
    next_frame_ts = time.perf_counter()

    try:
        while True:
            if not cap.isOpened():
                print("[ERROR] Could not open camera.")
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Blank frame grabbed.")
                break

            pil_img = lanefinder._preprocess(frame)

            if lanefinder._interpreter is not None:
                start_time = time.perf_counter()
                set_input(lanefinder._interpreter, pil_img)
                lanefinder._interpreter.invoke()
                duration_ms = (time.perf_counter() - start_time) * 1000
                output = lanefinder._interpreter.get_tensor(
                    lanefinder._interpreter.get_output_details()[0]['index']
                )
                frame_result = lanefinder._postprocess(output, frame.copy())
                print(f"[INFO] Inference time: {duration_ms:.2f} ms")
            else:
                frame_result = cv2.putText(
                    cv2.resize(frame.copy(), lanefinder._output_shape),
                    'TPU not detected!',
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2
                )

            success, buf = cv2.imencode(".jpg", frame_result, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                print("[WARNING] JPEG encoding failed.")
                continue

            # => send base64 string
            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            await websocket.send(b64)

            # 30 FPS pacing
            next_frame_ts += FRAME_INTERVAL
            delay = next_frame_ts - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_frame_ts = time.perf_counter()
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Client disconnected.")
    finally:
        cap.release()
        lanefinder.destroy()
        print("[INFO] Server shutdown.")


async def main():
    async with websockets.serve(video_stream,"0.0.0.0",8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
