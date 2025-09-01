import cv2
cap = cv2.VideoCapture(0)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera resolution: {cam_width}x{cam_height}")
import os
import json
import threading
import time
import glob
import re
import subprocess
from datetime import datetime
import yaml
from flask import Flask, Response, send_file, request, make_response, abort, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from image.warning import *
from image.processing import *

from tflite_runtime.interpreter import Interpreter
from pycoral.adapters import common, detect


from inference import Lanefinder

# ---- State load/save (unchanged) ----
def load_state():
    default = {"lanefinder_enabled": False,
               "recording_enabled": True,
               "detection_enabled": False,
               "camera_offset_bias": 0}
    if not os.path.isfile(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return {"lanefinder_enabled": bool(data.get("lanefinder_enabled", False)),
                "recording_enabled": bool(data.get("recording_enabled", True)),
                "detection_enabled": bool(data.get("detection_enabled", False)),
                "camera_offset_bias": int(data.get("camera_offset_bias", 0))}
    except Exception as e:
        print(f"[WARN] Can not read {STATE_FILE}: {e}")
        return default

def save_state(lanefinder_enabled: bool, recording_enabled: bool, detection_enabled: bool, camera_offset_bias: int):
    data = {"lanefinder_enabled": bool(lanefinder_enabled),
            "recording_enabled": bool(recording_enabled),
            "detection_enabled": bool(detection_enabled),
            "camera_offset_bias": int(camera_offset_bias)}
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Can not write {STATE_FILE}: {e}")

# ---- Object Detection helpers (ADDED) ----
def load_labels(path):
    label_map = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, name = parts
                label_map[int(idx)] = name
    return label_map

# ---- Flask + SocketIO ----
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---- Read config & init Lanefinder (unchanged) ----
def read_config():
    with open('config.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

# Constants
config = read_config()
STATE_FILE      = "state.json"
MODEL_PATH_OD   = config['od_model_path']
LABELS_PATH_OD  = config['od_labels']
# VEHICLE_CLASSES = {'car','bus','truck','motorcycle','traffic light','person','bicycle'}

state = load_state()
lanefinder_enabled = state["lanefinder_enabled"]
recording_enabled = state["recording_enabled"]
detection_enabled = state["detection_enabled"]
camera_offset_bias = state["camera_offset_bias"]
save_state(lanefinder_enabled, recording_enabled, detection_enabled, camera_offset_bias)
print(f"[INFO] Lanefinder state loaded: {lanefinder_enabled}")
print(f"[INFO] Recording state loaded: {recording_enabled}")
print(f"[INFO] Detection state loaded: {detection_enabled}")

lanefinder = Lanefinder(
    model_path=config['model_path'],
    input_shape=config['input_shape'],
    output_shape=tuple(config['output_shape']),
    quant=config['quantization'],
    dequant=config['dequantization']
)

# ---- Recording setup (unchanged) ----
results_folder = "video_recording"
thumbs_folder  = "thumbnails"
max_videos     = 30
segment_secs   = 60
recording_queue = []
recording_lock = threading.Lock()
last_written_time = time.time()
fps = 20
frame_interval = 1.0 / fps

os.makedirs(results_folder, exist_ok=True)
os.makedirs(thumbs_folder, exist_ok=True)

video_index      = 0
writer           = None
is_new_segment   = True
next_rotation_time = time.time() + segment_secs
stop_requested     = False

def recording_worker():
    global writer, is_new_segment, next_rotation_time, video_index, last_written_time

    while True:
        if not recording_enabled or writer is None:
            time.sleep(0.05)
            continue

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        # timestamp overlay
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, (255,255,255)
        h, w = frame.shape[:2]
        (tw, th_), _ = cv2.getTextSize(now_str, font, fs, th)
        cv2.rectangle(frame, (5, h - th_ - 15), (5 + tw + 5, h - 5), (0,0,0), cv2.FILLED)
        cv2.putText(frame, now_str, (10, h-10), font, fs, col, th, cv2.LINE_AA)
        # Ghi thumbnail nếu cần
        if is_new_segment:
            thumb_full = os.path.join(thumbs_folder, f"img_{video_index:02d}.jpg")
            cv2.imwrite(thumb_full, frame)
            print(f"[INFO] Saved thumbnail: {thumb_full}")
            is_new_segment = False

        # Ghi hình đều đặn theo FPS
        if now - last_written_time >= frame_interval:
            writer.write(frame)
            last_written_time = now

        # Kiểm tra xoay video segment
        if now >= next_rotation_time:
            avi_name = f"video_{video_index:02d}.avi"
            avi_full = os.path.join(results_folder, avi_name)
            threading.Thread(target=transcode_worker, args=(avi_full,), daemon=True).start()
            video_index = (video_index + 1) % max_videos
            open_new_writer()
            next_rotation_time = now + segment_secs

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
    global writer, video_index, is_new_segment
    if not recording_enabled or stop_requested:
        if writer is not None:
            writer.release()
            writer = None
        return
    if writer is not None:
        writer.release()
    avi_name = f"video_{video_index:02d}.avi"
    avi_full = os.path.join(results_folder, avi_name)
    if os.path.exists(avi_full):
        os.remove(avi_full)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20
    writer = cv2.VideoWriter(avi_full, fourcc, fps, (cam_width, cam_height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open VideoWriter for {avi_full}")
    print(f"[INFO] Now recording into: {avi_full}")
    is_new_segment = True

open_new_writer()

# ---- Lane detection (unchanged) ----
def detect_lanes(frame):
    pil_img = lanefinder._preprocess(frame)
    if lanefinder._interpreter is not None:
        t0 = time.perf_counter()
        common.set_input(lanefinder._interpreter, pil_img)
        lanefinder._interpreter.invoke()
        duration_ms = (time.perf_counter() - t0) * 1000
        output = lanefinder._interpreter.get_tensor(
            lanefinder._interpreter.get_output_details()[0]['index']
        )
        frame_result = lanefinder._postprocess(output, frame.copy(), camera_offset_bias)
        print(f"[INFO] Lane TPU time: {duration_ms:.2f} ms")
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
    return frame_result

# ---- Shared frames for threading (ADDED) ----
latest_frame    = None
detection_frame = None
frame_lock      = threading.Lock()
frame_count     = 0

# ---- Thread: grab camera frames (ADDED) ----
def camera_capture():
    global latest_frame
    while True:
        ret, frm = cap.read()
        if not ret:
            # time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frm.copy()
        # time.sleep(0.01)

def detection_worker():
    global detection_frame, frame_count
    interpreter = Interpreter(MODEL_PATH_OD)
    interpreter.allocate_tensors()
    labels_od = load_labels(LABELS_PATH_OD)
    while True:
        
        if not detection_enabled:
            with frame_lock:
                detection_frame = None
            time.sleep(0.1)
            continue
        with frame_lock:
            src = latest_frame.copy() if latest_frame is not None else None
        if src is None:
            time.sleep(0.01)
            continue
        frame_count += 1
        if frame_count % 3 == 0:
            _, scale = common.set_resized_input(
                interpreter,
                (src.shape[1], src.shape[0]),
                lambda size: cv2.resize(src, size)
            )
            t0 = time.perf_counter()
            interpreter.invoke()
            od_time = (time.perf_counter() - t0) * 1000
            # print(f"[INFO] OD time: {od_time:.2f} ms")
            dets = detect.get_objects(interpreter,
                                      score_threshold=0.5,
                                      image_scale=scale)
            out = src.copy()
            draw_detections(out, dets, labels_od)
            with frame_lock:
                detection_frame = out
        time.sleep(0.005)

# ---- MJPEG generator (modified) ----
def mjpeg_generator():
    global lanefinder_enabled, recording_enabled, stop_requested
    global video_index, writer, is_new_segment, next_rotation_time

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue

        # timestamp overlay
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, (255,255,255)
        h, w = frame.shape[:2]
        (tw, th_), _ = cv2.getTextSize(now_str, font, fs, th)
        cv2.rectangle(frame, (5, h - th_ - 15), (5 + tw + 5, h - 5), (0,0,0), cv2.FILLED)
        cv2.putText(frame, now_str, (10, h-10), font, fs, col, th, cv2.LINE_AA)

        orig_frame = frame.copy()

        # lane overlay
        if lanefinder_enabled:
            stream_frame = detect_lanes(frame)
        else:
            # stream_frame = frame.copy()
            stream_frame = cv2.resize(frame.copy(), lanefinder._output_shape)

        # object detection overlay
        with frame_lock:
            det = detection_frame.copy() if detection_frame is not None else None
        if det is not None:
            det = cv2.resize(det,(stream_frame.shape[1], stream_frame.shape[0]),interpolation=cv2.INTER_LINEAR)
            stream_frame = cv2.addWeighted(stream_frame, 0.8, det, 0.2, 0)


        # handle stopRequested and recording (unchanged logic)
        if stop_requested and writer is not None:
            avi_name = f"video_{video_index:02d}.avi"
            avi_full = os.path.join(results_folder, avi_name)
            threading.Thread(target=transcode_worker, args=(avi_full,), daemon=True).start()
            writer.release()
            writer = None
            recording_enabled = False
            stop_requested    = False

        # if recording_enabled and writer is not None:
        #     if is_new_segment:
        #         thumb_full = os.path.join(thumbs_folder, f"img_{video_index:02d}.jpg")
        #         cv2.imwrite(thumb_full, orig_frame)
        #         print(f"[INFO] Saved thumbnail: {thumb_full}")
        #         is_new_segment = False

        #     writer.write(orig_frame)
        #     now = time.time()
        #     if now >= next_rotation_time:
        #         avi_name = f"video_{video_index:02d}.avi"
        #         avi_full = os.path.join(results_folder, avi_name)
        #         threading.Thread(target=transcode_worker, args=(avi_full,), daemon=True).start()
        #         video_index = (video_index + 1) % max_videos
        #         open_new_writer()
        #         next_rotation_time += segment_secs
        #         if next_rotation_time < now:
        #             next_rotation_time = now + segment_secs

        # yield MJPEG
        ret2, buf = cv2.imencode('.jpg', stream_frame)
        h, w = stream_frame.shape[:2]
        # print(f"[INFO] Stream frame resolution: {w}x{h}")
        if not ret2:
            continue
        jpg = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.01)

    if writer:
        writer.release()

# ---- Flask routes (unchanged) ----
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
    safe = os.path.basename(filename)
    path = os.path.join(results_folder, safe)
    if not os.path.isfile(path):
        abort(404)
    size = os.path.getsize(path)
    range_hdr = request.headers.get('Range', None)
    if not range_hdr:
        resp = make_response(open(path, 'rb').read())
        resp.headers.add('Content-Length', str(size))
        resp.headers.add('Content-Type', 'video/mp4')
        resp.headers.add('Accept-Ranges', 'bytes')
        return resp
    m = re.match(r'bytes=(\d+)-(\d*)', range_hdr)
    if not m:
        abort(416)
    byte_s, byte_e = int(m.group(1)), m.group(2)
    byte_e = int(byte_e) if byte_e else size - 1
    if byte_s > byte_e or byte_e >= size:
        abort(416)
    length = byte_e - byte_s + 1
    with open(path, 'rb') as f:
        f.seek(byte_s)
        data = f.read(length)
    rv = make_response(data, 206)
    rv.headers.add('Content-Range', f'bytes {byte_s}-{byte_e}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    rv.headers.add('Content-Type', 'video/mp4')
    return rv

@app.route('/thumbnails/<filename>')
def serve_thumbnail(filename):
    safe_name = os.path.basename(filename)
    full_path = os.path.join(thumbs_folder, safe_name)
    if not os.path.isfile(full_path):
        abort(404)
    
    response = send_file(full_path, mimetype="image/jpeg", conditional=True)
    # Prevent caching of thumbnails
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/gallery')
def gallery_index():
    base = request.host_url.rstrip('/')
    entries = []
    thumbs = glob.glob(os.path.join(thumbs_folder, "img_*.jpg"))
    thumbs.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    for thumb in thumbs:
        fname = os.path.basename(thumb)
        m = re.match(r'img_([0-9]+)\.jpg$', fname)
        if not m: continue
        idx = m.group(1)
        mp4 = f"video_{idx}.mp4"
        mp4_path = os.path.join(results_folder, mp4)
        if not os.path.isfile(mp4_path):
            continue

        mod_time = int(os.path.getmtime(thumb))
        entries.append({
            "id": idx,
            "thumbnail_url": f"{base}/thumbnails/{fname}?t={mod_time}",
            "video_url":     f"{base}/videos/{mp4}"
        })
    return Response(json.dumps(entries), mimetype='application/json')

@app.route('/gallery/delete', methods=['POST'])
def delete_videos():
    try:
        data = request.get_json()
        ids = data.get('ids', [])

        print(ids)


        if not ids or not isinstance(ids, list):
            return jsonify({'error': 'Invalid or missing "ids"'}), 400

        deleted = []
        for vid in ids:
            video_pattern_avi = f"video_recording/video_{vid}.avi"
            video_pattern_mp4 = f"video_recording/video_{vid}.mp4"
            thumb_path = f"thumbnails/img_{vid}.jpg"

            try:
                subprocess.run(['rm', '-rf', video_pattern_avi], check=True, shell=False)
                subprocess.run(['rm', '-rf', video_pattern_mp4], check=True, shell=False)
                subprocess.run(['rm', '-f', thumb_path], check=True, shell=False)
                deleted.append(vid)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to delete {vid}: {e}")

        return jsonify({'deleted': deleted})

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        return jsonify({'error': 'Server error'}), 500

@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")
    emit('initialState', {
        'lanefinder_enabled': lanefinder_enabled,
        'recording_enabled': recording_enabled,
        'camera_offset_bias': camera_offset_bias
    })

@socketio.on('toggleLanefinder')
def on_toggle(data):
    global lanefinder_enabled
    lanefinder_enabled = bool(data.get('enabled', False))
    print(f"[INFO] Lanefinder toggled → {lanefinder_enabled}")
    save_state(lanefinder_enabled, recording_enabled, detection_enabled, camera_offset_bias)
    emit('updateLanefinder', {'lanefinder_enabled': lanefinder_enabled})

@socketio.on('toggleDetection')
def on_toggle(data):
    global detection_enabled
    detection_enabled = bool(data.get('enabled', False))
    print(f"[INFO] Detection toggled → {detection_enabled}")
    save_state(lanefinder_enabled, recording_enabled, detection_enabled, camera_offset_bias)
    emit('updateDetection', {'detection_enabled': detection_enabled})

@socketio.on('stopRecording')
def on_stop_recording(data):
    global stop_requested, recording_enabled, next_rotation_time, video_index
    if not data.get('enabled', False):
        print("[INFO] stopRecording → finalizing current segment")
        stop_requested = True
        recording_enabled = False
    else:
        if recording_enabled:
            return
        print("[INFO] resumeRecording")
        recording_enabled = True
        stop_requested    = False
        save_state(lanefinder_enabled, recording_enabled, detection_enabled, camera_offset_bias)
        video_index = (video_index + 1) % max_videos
        next_rotation_time = time.time() + segment_secs
        open_new_writer()
    emit('updateRecording', {'recording_enabled': recording_enabled}, broadcast=True)

@socketio.on('updateCameraBias')
def handle_bias(data):
    global camera_offset_bias
    camera_offset_bias = data.get('bias', 0.0)
    save_state(lanefinder_enabled, recording_enabled, detection_enabled, camera_offset_bias)
    print(f"[UPDATE] Camera bias angle: {camera_offset_bias:.2f}°")
    emit('updateBias', {'camera_offset_bias':camera_offset_bias}, broadcast=True)

if __name__ == '__main__':
    # start background threads
    threading.Thread(target=camera_capture, daemon=True).start()
    threading.Thread(target=detection_worker, daemon=True).start()
    threading.Thread(target=recording_worker, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)