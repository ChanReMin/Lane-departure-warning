import cv2
cap = cv2.VideoCapture(0)
import socket
import time
import numpy as np
from inference import Lanefinder
from pycoral.adapters.common import set_input  # If you use pycoral in Lanefinder
import yaml


def read_config():
    """Reads the config file (config.yaml)."""
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def start_mjpeg_server(host='0.0.0.0', port=8080):
    """
    Creates a simple socket server that streams MJPEG frames to one client at a time.

    Returns the accepted client socket once a connection is made.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"[INFO] MJPEG server listening on http://{host}:{port}")

    client_socket, addr = server_socket.accept()
    print(f"[INFO] Client connected from: {addr}")

    # Send the HTTP headers for MJPEG streaming
    headers = (
        "HTTP/1.0 200 OK\r\n"
        "Server: PythonInferenceServer\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
    )
    client_socket.sendall(headers.encode("utf-8"))

    return client_socket, server_socket


def main():
    # --------------------------------------------------------------------------
    # 1) Read config & initialize inference model (Lanefinder)
    # --------------------------------------------------------------------------
    config = read_config()
    lanefinder = Lanefinder(
        model_path=config['model_path'],
        input_shape=config['input_shape'],
        output_shape=tuple(config['output_shape']),
        quant=config['quantization'],
        dequant=config['dequantization']
    )

    # --------------------------------------------------------------------------
    # 2) Open camera (USB camera index 0). Adjust index if needed.
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # 3) Start the MJPEG server. Wait for a client to connect (blocking).
    # --------------------------------------------------------------------------
    client_socket, server_socket = start_mjpeg_server(host="0.0.0.0", port=8080)

    # We’ll prepare a small function to send frames as JPEG data to the client.
    def send_mjpeg_frame(sock, frame_bytes):
        """
        Send a single frame (already JPEG-encoded) to client over MJPEG stream.
        """
        frame_header = (
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame_bytes)}\r\n\r\n"
        )
        sock.sendall(frame_header.encode("utf-8"))
        sock.sendall(frame_bytes)
        sock.sendall(b"\r\n")

    try:
        # ----------------------------------------------------------------------
        # 4) Continuously capture frames from camera, run inference, and send.
        # ----------------------------------------------------------------------
        while True:
            if not cap.isOpened():
                print("[ERROR] Could not open camera.")
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Blank frame grabbed.")
                break

            # Preprocess for TFLite model
            pil_img = lanefinder._preprocess(frame)

            # Inference
            if lanefinder._interpreter is not None:
                start_time = time.perf_counter()
                set_input(lanefinder._interpreter, pil_img)
                lanefinder._interpreter.invoke()
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Get output tensor
                output = lanefinder._interpreter.get_tensor(
                    lanefinder._interpreter.get_output_details()[0]['index']
                )

                # Postprocess result (draw lines on original frame copy, e.g.)
                frame_result = lanefinder._postprocess(output, frame.copy())
                print(f"[INFO] Inference time: {duration_ms:.2f} ms")
            else:
                # Fallback if TPU not detected
                frame_result = cv2.putText(
                    cv2.resize(frame.copy(), lanefinder._output_shape),
                    'TPU not detected!',
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2
                )

            # JPEG-encode the inference result in-memory
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, jpg_bytes = cv2.imencode(".jpg", frame_result, encode_param)
            if not success:
                print("[WARNING] JPEG encoding failed.")
                continue

            # Send to the connected browser/client
            send_mjpeg_frame(client_socket, jpg_bytes.tobytes())

    except BrokenPipeError:
        print("[INFO] Client disconnected.")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        # Cleanup
        cap.release()
        lanefinder.destroy()
        client_socket.close()
        server_socket.close()
        print("[INFO] Server shutdown.")


if __name__ == "__main__":
    main()
