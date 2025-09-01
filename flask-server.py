import cv2
# Mở camera 1 lần
cap = cv2.VideoCapture(0)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera resolution: {cam_width}x{cam_height}")
import os
import json
import time
import threading
import subprocess
from datetime import datetime

import yaml
from flask import Flask, Response, send_file, request, abort, make_response
from flask_socketio import SocketIO, emit
from inference import Lanefinder
from pycoral.adapters.common import set_input
import glob

# =========================
# I. CÀI ĐẶT BAN ĐẦU
# =========================
app = Flask(__name__)
# Ép dùng threading (không eventlet/gevent)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load config cho lanefinder
def read_config():
    with open('config.yaml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

config = read_config()
lanefinder = Lanefinder(
    model_path=config['model_path'],
    input_shape=config['input_shape'],
    output_shape=tuple(config['output_shape']),
    quant=config['quantization'],
    dequant=config['dequantization']
)


# Folder lưu video và thumbnails
results_folder = "video_recording"
thumbs_folder = "thumbnails"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(thumbs_folder, exist_ok=True)

# Tham số quay video
max_videos = 30
segment_secs = 60

# Biến toàn cục
STATE_FILE = "state.json"

latest_frame = None
frame_lock = threading.Lock()

lanefinder_enabled = False
recording_enabled = True
stop_requested = False

video_index = 0
writer = None
next_rotation_time = time.time() + segment_secs

# ============================
# II. HÀM LƯU/LOAD STATE
# ============================
def load_state():
    default = {"lanefinder_enabled": False, "recording_enabled": True}
    if not os.path.isfile(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return {
            "lanefinder_enabled": bool(data.get("lanefinder_enabled", False)),
            "recording_enabled": bool(data.get("recording_enabled", True))
        }
    except Exception as e:
        print(f"[WARN] Không đọc được {STATE_FILE}: {e}")
        return default

def save_state():
    global lanefinder_enabled, recording_enabled
    data = {
        "lanefinder_enabled": lanefinder_enabled,
        "recording_enabled": recording_enabled
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Không ghi được {STATE_FILE}: {e}")

state = load_state()
lanefinder_enabled = state["lanefinder_enabled"]
recording_enabled = state["recording_enabled"]
print(f"[INFO] Lanefinder state: {lanefinder_enabled}")
print(f"[INFO] Recording state: {recording_enabled}")

# ================================
# III. HÀM LÀM LANEFINDER (giống cũ)
# ================================
def detect_lanes(frame):
    pil_img = lanefinder._preprocess(frame)
    if lanefinder._interpreter is not None:
        t0 = time.perf_counter()
        set_input(lanefinder._interpreter, pil_img)
        lanefinder._interpreter.invoke()
        duration_ms = (time.perf_counter() - t0) * 1000
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
        print(f"[DEBUG] Fallback (resized): {frame_result.shape}")
    return frame_result

# ============================
# IV. HÀM QUAY VIDEO (recorder)
# ============================
def open_new_writer():
    global writer, video_index, next_rotation_time

    # Đóng writer cũ
    if writer is not None:
        writer.release()
        writer = None

    if not recording_enabled or stop_requested:
        return

    avi_name = f"video_{video_index:02d}.avi"
    avi_full_path = os.path.join(results_folder, avi_name)
    if os.path.exists(avi_full_path):
        os.remove(avi_full_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20
    writer = cv2.VideoWriter(avi_full_path, fourcc, fps, (cam_width, cam_height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Không mở được VideoWriter cho {avi_full_path}")

    print(f"[INFO] Bắt đầu ghi segment: {avi_full_path}")
    next_rotation_time = time.time() + segment_secs

def transcode_worker(avi_full_path):
    try:
        print(f"[INFO] Transcode start cho {avi_full_path}")
        mp4_full_path = avi_full_path.rsplit('.', 1)[0] + '.mp4'
        cmd = [
            'ffmpeg', '-y', '-i', avi_full_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            mp4_full_path
        ]
        subprocess.run(cmd, check=True)
        print(f"[INFO] Transcode xong: {mp4_full_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg lỗi {avi_full_path}: {e}")

def recorder_thread():
    """
    Chạy độc lập để ghi frame mới nhất vào AVI, rotate, generate thumbnail, transcode.
    """
    global latest_frame, video_index, writer, stop_requested, next_rotation_time, recording_enabled

    first_frame = True
    while True:
        # Nếu có yêu cầu stop thì finalize segment hiện tại
        if stop_requested and writer is not None:
            avi_name = f"video_{video_index:02d}.avi"
            avi_full = os.path.join(results_folder, avi_name)
            print(f"[INFO] Finalize ngay {avi_full} (stop_requested)")
            threading.Thread(
                target=transcode_worker,
                args=(avi_full,),
                daemon=True
            ).start()
            writer.release()
            writer = None
            recording_enabled = False
            stop_requested = False
            first_frame = True

        if recording_enabled:
            with frame_lock:
                frame_copy = None if latest_frame is None else latest_frame.copy()
            if frame_copy is not None:
                if first_frame:
                    thumb_name = f"img_{video_index:02d}.jpg"
                    thumb_full = os.path.join(thumbs_folder, thumb_name)
                    cv2.imwrite(thumb_full, frame_copy)
                    print(f"[INFO] Lưu thumbnail: {thumb_full}")
                    first_frame = False

                writer.write(frame_copy)
                # Check rotate:
                now = time.time()
                if now >= next_rotation_time:
                    avi_name = f"video_{video_index:02d}.avi"
                    avi_full = os.path.join(results_folder, avi_name)
                    print(f"[INFO] Đủ {segment_secs}s → đóng {avi_full}")
                    threading.Thread(
                        target=transcode_worker,
                        args=(avi_full,),
                        daemon=True
                    ).start()

                    video_index = (video_index + 1) % max_videos
                    open_new_writer()
                    first_frame = True
                    next_rotation_time = now + segment_secs

        # Giảm độ “điên” CPU
        time.sleep(0.02)

# ============================
# V. HÀM ĐỌC CAMERA (grabber)
# ============================
def grabber_thread():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
        # Nếu muốn đọc chậm lại, có thể sleep, nhưng để min delay
        # time.sleep(0.001)

# ============================
# VI. HÀM STREAM MJPEG (generator)
# ============================
def mjpeg_generator():
    global lanefinder_enabled
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        # Vẽ timestamp
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

        if lanefinder_enabled:
            processed = detect_lanes(frame)
            to_encode = processed
        else:
            to_encode = cv2.resize(frame, lanefinder._output_shape)

        ret2, buf = cv2.imencode('.jpg', to_encode)
        if not ret2:
            continue
        jpg = buf.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        )

# ============================
# VII. ROUTES CHÍNH
# ============================
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
        abort(404)
    return send_file(full_path, mimetype=mimetype, conditional=True)

@app.route('/videos/<filename>')
def serve_video(filename):
    safe_name = os.path.basename(filename)
    video_path = os.path.join(results_folder, safe_name)
    if not os.path.isfile(video_path):
        return abort(404)

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get('Range', None)
    if not range_header:
        response = make_response(open(video_path, 'rb').read())
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Content-Type', 'video/mp4')
        response.headers.add('Accept-Ranges', 'bytes')
        return response

    m = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not m:
        return abort(416)

    byte_s = int(m.group(1))
    byte_e = int(m.group(2)) if m.group(2) else file_size - 1
    if byte_s >= file_size or byte_e >= file_size or byte_s > byte_e:
        return abort(416)
    length = byte_e - byte_s + 1
    with open(video_path, 'rb') as f:
        f.seek(byte_s)
        data = f.read(length)
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

# ============================
# VIII. SOCKET.IO EVENTS
# ============================
@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")
    emit('initialState', {
        'lanefinder_enabled': lanefinder_enabled,
        'recording_enabled': recording_enabled
    })

@socketio.on('toggleLanefinder')
def on_toggle(data):
    global lanefinder_enabled
    lanefinder_enabled = bool(data.get('enabled', False))
    print(f"[INFO] Lanefinder toggled → {lanefinder_enabled}")
    save_state()
    emit('updateLanefinder', {'lanefinder_enabled': lanefinder_enabled})

@socketio.on('stopRecording')
def on_stop_recording(data):
    global stop_requested, recording_enabled, video_index
    # Nếu data["enabled"] == True → STOP
    if data.get('enabled', False):
        print("[INFO] stopRecording = True → request stop")
        stop_requested = True
    else:
        # START/RESUME
        if recording_enabled:
            return
        print("[INFO] stopRecording = False → Resume")
        recording_enabled = True
        stop_requested = False
        video_index = (video_index + 1) % max_videos
        open_new_writer()
    save_state()
    emit('updateRecording', {'recording_enabled': recording_enabled}, broadcast=True)

@socketio.on('updateCameraBias')
def handle_bias(data):
    global camera_offset_angle_bias
    camera_offset_angle_bias = data.get('bias', 0.0)
    print(f"[UPDATE] Camera bias angle: {camera_offset_angle_bias:.2f} degrees")

# ============================
# IX. KHỞI ĐỘNG CHƯƠNG TRÌNH
# ============================
if __name__ == '__main__':
    # 1. Start grabber thread
    t_grabber = threading.Thread(target=grabber_thread, daemon=True)
    t_grabber.start()

    # 2. Mở writer nếu cần
    if recording_enabled:
        open_new_writer()

    # 3. Start recorder thread
    t_recorder = threading.Thread(target=recorder_thread, daemon=True)
    t_recorder.start()

    # 4. Chạy Flask-SocketIO (threading mode)
    socketio.run(app, host='0.0.0.0', port=5000)
