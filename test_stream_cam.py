#!/usr/bin/env python3
"""
Very-lightweight MJPEG streaming server for <img src="http://<host>:8080" />.
Tested on Python 3.8+ with OpenCV ≥ 4.  No third-party web framework required.
"""

import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import signal
import sys

# ──────────────────────────────── Configuration ────────────────────────────────
PORT = 8080
JPEG_QUALITY = 80            # Same as your C++ vector<int>{cv::IMWRITE_JPEG_QUALITY, 80}
BOUNDARY = b"--frame\r\n"    # Multipart boundary
CAMERA_INDEX = 0             # /dev/video0 on Linux / default webcam on Windows

# ─────────────────────────── Camera capture thread ─────────────────────────────
class Camera:
    """Grabs frames continuously in a background thread so HTTP can read quickly."""
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # Pre-encode to JPEG bytes to avoid doing it for each client
            ok, buf = cv2.imencode(".jpg", frame, self.encode_params)
            if ok:
                with self.lock:
                    self.frame = buf.tobytes()

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()

camera = Camera(CAMERA_INDEX)

# ───────────────────────────── HTTP handler class ──────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve a single multipart MJPEG stream."""
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Server", "Python-MJPEG-Server")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # --frame
                self.wfile.write(BOUNDARY)
                # headers for each JPEG part
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n"
                                 .encode('utf-8'))
                # raw JPEG bytes
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                # quick flush to keep latency low
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            # Client closed the connection
            pass
        except Exception as e:
            print("Streaming error:", e)

    # Silence the default noisy log lines (optional)
    def log_message(self, fmt, *args):
        return

# ───────────────────────────── Entrypoint / server ─────────────────────────────
def run():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MJPEGHandler)

    # Graceful Ctrl-C handling
    def shutdown(sig, frame):
        print("\nStopping server …")
        server.shutdown()
        camera.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print(f"MJPEG server listening on http://<this-host>:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    run()
