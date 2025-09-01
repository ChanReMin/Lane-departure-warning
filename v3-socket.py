import cv2
cap = cv2.VideoCapture(0)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera resolution: {cam_width}x{cam_height}")

import json
import os

STATE_FILE = "state.json"

def load_state():
    default = {"lanefinder_enabled": False,
               "recording_enabled": True}
    if not os.path.isfile(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        # Chỉ lấy đúng trường "lanefinder_enabled", nếu thiếu thì dùng mặc định
        return {"lanefinder_enabled": bool(data.get("lanefinder_enabled", False)),
                "recording_enabled": bool(data.get("recording_enabled", True))}
    except Exception as e:
        print(f"[WARN] Can not read {STATE_FILE}: {e}")
        return default

def save_state(lanefinder_enabled: bool, recording_enabled: bool):
    data = {"lanefinder_enabled": bool(lanefinder_enabled),
            "recording_enabled": bool(recording_enabled)}
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Can not write {STATE_FILE}: {e}")

from flask import Flask, Response, send_file, request, make_response, abort
from flask_socketio import SocketIO, emit
import numpy as np
from inference import Lanefinder
from pycoral.adapters.common import set_input
import yaml
import time
import glob
import re
import threading
import subprocess
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def read_config():
    """Reads the config file (config.yaml)."""
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Load saved state
state = load_state()
lanefinder_enabled = state["lanefinder_enabled"]
recording_enabled = state["recording_enabled"]
# Force recording_enabled = True on startup
recording_enabled = True
save_state(lanefinder_enabled, recording_enabled)
print(f"[INFO] Lanefinder state loaded: {lanefinder_enabled}")
print(f"[INFO] Recording state loaded: {recording_enabled}")

config = read_config()
lanefinder = Lanefinder(
    model_path=config['model_path'],
    input_shape=config['input_shape'],
    output_shape=tuple(config['output_shape']),
    quant=config['quantization'],
    dequant=config['dequantization']
)

# Recording Parameters
results_folder = "video_recording"
thumbs_folder = "thumbnails"
max_videos = 30
segment_secs = 60

# Ensure the results and thumbnails folders exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(thumbs_folder, exist_ok=True)

# Globals to track current recording segment
video_index = 0
writer = None
is_new_segment = True
next_rotation_time = time.time() + segment_secs
stop_requested = False

# Shared frame buffer
current_frame = None
frame_lock = threading.Lock()

def transcode_worker(avi_full_path):
    try:
        print(f"[INFO] Transcoding started for {avi_full_path}")
        mp4_full_path = avi_full_path.rsplit('.', 1)[0] + '.mp4'
        cmd = [
            'ffmpeg', '-y', '-i', avi_full_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            mp4_full_path
        ]
        subprocess.run(cmd, check=True)
        print(f"[INFO] Transcoding finished: {mp4_full_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed on {avi_full_path}: {e}")

def open_new_writer():
    """
    Closes any existing writer and opens a new one for the next index.
    Uses MJPG codec at 20 fps, with frame size = (cam_width, cam_height).
    """
    global writer, video_index, is_new_segment, next_rotation_time

    if not recording_enabled or stop_requested:
        # Nếu không ghi, đảm bảo writer đóng
        if writer is not None:
            writer.release()
            writer = None
        return

    # Release old writer nếu còn mở
    if writer is not None:
        writer.release()

    # Filename: video_{index}.avi
    avi_name = f"video_{video_index:02d}.avi"
    avi_full_path = os.path.join(results_folder, avi_name)

    if os.path.exists(avi_full_path):
        os.remove(avi_full_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20
    writer = cv2.VideoWriter(avi_full_path, fourcc, fps, (cam_width, cam_height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open VideoWriter for {avi_full_path}")

    print(f"[INFO] Now recording into: {avi_full_path}")
    is_new_segment = True
    next_rotation_time = time.time() + segment_secs

class FrameGrabberThread(threading.Thread):
    """
    Thread này chuyên đọc frame từ camera, cập nhật current_frame.
    """
    def __init__(self, cap, lock):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = lock

    def run(self):
        global current_frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                current_frame = frame.copy()
            time.sleep(0.01)

class RecorderThread(threading.Thread):
    """
    Thread này chuyên ghi video dựa trên current_frame.
    """
    def __init__(self, lock):
        super().__init__(daemon=True)
        self.lock = lock

    def run(self):
        global recording_enabled, stop_requested, writer, video_index, is_new_segment, next_rotation_time

        while True:
            if recording_enabled:
                # Nếu writer chưa mở, mở new writer
                if writer is None:
                    open_new_writer()

                now = time.time()
                frame = None
                with self.lock:
                    if current_frame is not None:
                        frame = current_frame.copy()

                if frame is not None and writer is not None:
                    # Lưu thumbnail nếu segment mới
                    if is_new_segment:
                        thumb_name = f"img_{video_index:02d}.jpg"
                        thumb_full = os.path.join(thumbs_folder, thumb_name)
                        cv2.imwrite(thumb_full, frame)
                        print(f"[INFO] Saved thumbnail: {thumb_full}")
                        is_new_segment = False

                    # Ghi frame gốc (resize nếu cần)
                    resized = cv2.resize(frame, (cam_width, cam_height))
                    writer.write(resized)

                # Kiểm tra time rotate
                if now >= next_rotation_time and writer is not None:
                    avi_name = f"video_{video_index:02d}.avi"
                    avi_full = os.path.join(results_folder, avi_name)
                    print(f"[INFO] 60s reached → closing {avi_full}")
                    writer.release()
                    threading.Thread(
                        target=transcode_worker,
                        args=(avi_full,),
                        daemon=True
                    ).start()

                    video_index = (video_index + 1) % max_videos
                    open_new_writer()

                # Nếu có stop_requested
                if stop_requested and writer is not None:
                    avi_name = f"video_{video_index:02d}.avi"
                    avi_full = os.path.join(results_folder, avi_name)
                    print(f"[INFO] stopRecording: finalizing {avi_full} immediately.")
                    writer.release()
                    threading.Thread(
                        target=transcode_worker,
                        args=(avi_full,),
                        daemon=True
                    ).start()
                    writer = None
                    recording_enabled = False
                    stop_requested = False

                time.sleep(0.01)
            else:
                time.sleep(0.1)

def detect_lanes(frame):
    pil_img = lanefinder._preprocess(frame)
    if lanefinder._interpreter is not None:
        t0 = time.perf_counter()
        set_input(lanefinder._interpreter, pil_img)
        lanefinder._interpreter.invoke()
        duration_ms = (time.perf_counter() - t0) * 1000

        # Get output tensor
        output = lanefinder._interpreter.get_tensor(
            lanefinder._interpreter.get_output_details()[0]['index']
        )

        # Postprocess result (draw lines on original frame copy)
        frame_result = lanefinder._postprocess(output, frame.copy())
        print(f"[INFO] Inference time: {duration_ms:.2f} ms")
    else:
        # Fallback nếu TPU không detect được
        frame_result = cv2.putText(
            cv2.resize(frame.copy(), lanefinder._output_shape),
            'TPU not detected!',
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2
        )
        print(f"[DEBUG] Fallback frame size (resized): {frame_result.shape}")
    return frame_result

def mjpeg_generator():
    """
    Generator MJPEG: lấy current_frame, vẽ timestamp + detect lanes, encode và yield.
    """
    global lanefinder_enabled

    while True:
        frame = None
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        # Vẽ timestamp lên frame
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        color     = (255, 255, 255)

        h, w = frame.shape[:2]
        org = (10, h - 10)

        (text_w, text_h), _ = cv2.getTextSize(now_str, font, fontScale, thickness)
        cv2.rectangle(
            frame,
            (org[0] - 5, org[1] - text_h - 5),
            (org[0] + text_w + 5, org[1] + 5),
            (0, 0, 0),
            thickness=cv2.FILLED
        )
        cv2.putText(frame, now_str, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Chọn khung cho streaming
        if lanefinder_enabled:
            frame_for_stream = detect_lanes(frame)
        else:
            frame_for_stream = cv2.resize(frame.copy(), lanefinder._output_shape)

        # Encode sang JPEG và yield
        ret2, buf = cv2.imencode('.jpg', frame_for_stream)
        if not ret2:
            time.sleep(0.01)
            continue

        jpg = buf.tobytes()
        try:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            )
        except GeneratorExit:
            # Khi client ngắt kết nối, dừng generator
            print("[Stream ] Client disconnected, stopping mjpeg_generator.")
            return
        except Exception as e:
            print(f"[Stream ] Lỗi khi yield frame: {e}")
            return

        # Giới hạn fps streaming ~20fps
        time.sleep(1.0 / 20.0)

@app.route('/')
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def serve_file_from_folder(folder, filename, mimetype=None):
    safe_name = os.path.basename(filename)
    full_path = os.path.join(folder, safe_name)
    if not os.path.isfile(full_path):
        print("not found")
    return send_file(full_path, mimetype=mimetype, conditional=True)

@app.route('/videos/<filename>')
def serve_video(filename):
    # 1. Xác định đường dẫn file
    safe_name = os.path.basename(filename)
    video_path = os.path.join(results_folder, safe_name)
    if not os.path.isfile(video_path):
        return abort(404)

    # 2. Lấy kích thước file (bytes)
    file_size = os.path.getsize(video_path)

    # 3. Đọc header "Range" (nếu có)
    range_header = request.headers.get('Range', None)
    if not range_header:
        # Nếu client không yêu cầu range, trả toàn bộ file (status 200)
        response = make_response(open(video_path, 'rb').read())
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Content-Type', 'video/mp4')
        response.headers.add('Accept-Ranges', 'bytes')
        return response

    # 4. Nếu có Range, parse giá trị
    m = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not m:
        return abort(416)

    byte_s = int(m.group(1))
    byte_e = m.group(2)
    if byte_e:
        byte_e = int(byte_e)
    else:
        byte_e = file_size - 1

    # 5. Kiểm tra tính hợp lệ của range
    if byte_s >= file_size or byte_e >= file_size or byte_s > byte_e:
        return abort(416)

    # 6. Tính độ dài của phần dữ liệu cần trả
    length = byte_e - byte_s + 1

    # 7. Đọc đúng đoạn bytes từ file
    with open(video_path, 'rb') as f:
        f.seek(byte_s)
        data = f.read(length)

    # 8. Tạo response với status 206 (Partial Content)
    rv = make_response(data, 206)
    rv.headers.add('Content-Range', f'bytes {byte_s}-{byte_e}/{file_size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    rv.headers.add('Content-Type', 'video/mp4')
    return rv

@app.route('/thumbnails/<filename>')
def serve_thumbnail(filename):
    return serve_file_from_folder(thumbs_folder, filename, mimetype="image/jpeg")

@app.route('/gallery')
def gallery_index():
    base = request.host_url.rstrip('/')
    entries = []
    pattern = os.path.join(thumbs_folder, "img_*.jpg")

    for thumb_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(thumb_path)
        m = re.match(r'img_([0-9]+)\.jpg$', fname)
        if not m:
            continue

        idx_str = m.group(1)
        mp4_name = f"video_{idx_str}.mp4"
        mp4_full = os.path.join(results_folder, mp4_name)
        if not os.path.isfile(mp4_full):
            continue

        mtime = os.path.getmtime(mp4_full)
        entries.append({
            "id": idx_str,
            "thumbnail_url": f"{base}/thumbnails/img_{idx_str}.jpg",
            "video_url":     f"{base}/videos/{mp4_name}",
            "mtime": mtime
        })

    entries.sort(key=lambda e: e["mtime"], reverse=True)

    for e in entries:
        del e["mtime"]
    return Response(json.dumps(entries), mimetype='application/json')

@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")
    emit('initialState', {
        'lanefinder_enabled': lanefinder_enabled,
        'recording_enabled': recording_enabled
    })

@socketio.on('toggleLanefinder')
def on_toggle(data):
    """
    Receives: { "enabled": true|false }
    """
    global lanefinder_enabled, recording_enabled
    
    new_val = bool(data.get('enabled', False))
    lanefinder_enabled = new_val
    print(f"[INFO] Lanefinder toggled → {lanefinder_enabled}")

    save_state(lanefinder_enabled, recording_enabled)
    emit('updateLanefinder', {'lanefinder_enabled': lanefinder_enabled})

@socketio.on('stopRecording')
def on_stop_recording(data):
    """
    Client emits { enabled: true } when it wants to STOP recording.
    Client emits { enabled: false } when it wants to START (or resume) recording.
    """
    global stop_requested, recording_enabled, next_rotation_time, video_index, lanefinder_enabled

    if not data.get('enabled', False):
        # -- STOP RECORDING PATH --
        print("[INFO] Received stopRecording = True → finalize current and stop.")
        stop_requested = True
        recording_enabled = False
        # Mjpe g_generator không làm gì với việc stop; RecorderThread sẽ xử lý finalize
    else:
        # -- START/RESUME RECORDING PATH --
        if recording_enabled:
            # Đang chạy rồi, không làm gì
            return

        print("[INFO] Received stopRecording = False → Resume recording.")
        recording_enabled = True
        stop_requested = False
        save_state(lanefinder_enabled, recording_enabled)
        
        video_index = (video_index + 1) % max_videos
        next_rotation_time = time.time() + segment_secs
        # Writer sẽ được mở khi RecorderThread thấy recording_enabled=True
    emit('updateRecording', {'recording_enabled': recording_enabled}, broadcast=True)

@socketio.on('updateCameraBias')
def handle_bias(data):
    global camera_offset_angle_bias
    camera_offset_angle_bias = data.get('bias', 0.0)
    print(f"[UPDATE] Camera bias angle: {camera_offset_angle_bias:.2f} degrees")

if __name__ == '__main__':
    # Khởi chạy FrameGrabberThread và RecorderThread trước khi serve
    grabber = FrameGrabberThread(cap, frame_lock)
    grabber.start()

    recorder = RecorderThread(frame_lock)
    recorder.start()

    # Chạy Flask + SocketIO
    socketio.run(app, host='0.0.0.0', port=5000)
