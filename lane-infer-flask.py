import cv2
cap = cv2.VideoCapture(0)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera resolution: {cam_width}x{cam_height}")

from flask import Flask, Response, send_file, request
from flask_socketio import SocketIO
import numpy as np
from inference import Lanefinder
from pycoral.adapters.common import set_input
import yaml
import time
import os
import glob
import re
import json
import threading
import subprocess


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def read_config():
    """Reads the config file (config.yaml)."""
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
lanefinder_enabled = False  # shared flag

config = read_config()
lanefinder = Lanefinder(
    model_path=config['model_path'],
    input_shape=config['input_shape'],
    output_shape=tuple(config['output_shape']),
    quant=config['quantization'],
    dequant=config['dequantization']
)

#Recording Parameters
results_folder = "video_recording"
thumbs_folder = "thumbnails"
max_videos = 10
segment_secs = 60

# Ensure the results folder exists
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Globals to track current recording segment
video_index = 0
writer = None
is_new_segment = True

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
    Uses MJPG codec at 20 fps, with frame size = lanefinder._output_shape.
    """
    global writer, video_index, is_new_segment

    # Release old writer if it exists
    if writer is not None:
        writer.release()

    # **Use "video_{index}.avi" as the filename**:
    avi_name = f"video_{video_index:02d}.avi"
    avi_full_path = os.path.join(results_folder, avi_name)

    if os.path.exists(avi_full_path):
        os.remove(avi_full_path)

    # FourCC for MJPG
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20

    # Create new VideoWriter
    writer = cv2.VideoWriter(avi_full_path, fourcc, fps, (cam_width, cam_height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open VideoWriter for {filename}")

    print(f"[INFO] Started new video file: {avi_full_path}")
    start_time = time.time()  # reset the timer
    is_new_segment = True

def rotate_segment():
    """
    Called every `segment_secs` seconds by a Timer. 
    It:
      1. Hands off the just‐finished AVI to a background thread for transcoding.
      2. Increments `video_index` and opens a new writer immediately.
      3. Schedules itself to run again in `segment_secs` seconds.
    """
    global video_index

    # 1. Identify the AVI we just closed
    avi_name = f"video_{video_index:02d}.avi"
    avi_full_path = os.path.join(results_folder, avi_name)
    print(f"[INFO] Finished filling {avi_full_path} (60 s reached).")

    # Launch FFmpeg transcoding in background
    threading.Thread(target=transcode_worker, args=(avi_full_path,), daemon=True).start()

    # 2. Advance to next index (ring buffer) and start a new writer
    video_index = (video_index + 1) % max_videos
    open_new_writer()

    # 3. Schedule the next rotation in exactly segment_secs seconds
    threading.Timer(segment_secs, rotate_segment).start()
# Immediately open the first writer before streaming begins
open_new_writer()
threading.Timer(segment_secs, rotate_segment).start()

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
        print(f"[DEBUG] Fallback frame size (resized): {frame_result.shape}")
    return frame_result

def mjpeg_generator():
    global lanefinder_enabled, video_index, writer, is_new_segment
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        orig_frame = frame.copy()
        
        # If this is the very first frame of a new segment, save its thumbnail
        if is_new_segment:
            thumb_path = os.path.join(thumbs_folder, f"img_{video_index:02d}.jpg")
            cv2.imwrite(thumb_path, orig_frame)
            print(f"[INFO] Saved thumbnail: {thumb_path}")
            is_new_segment = False
        
        if lanefinder_enabled:
            frame_processed = detect_lanes(frame)
        else:
            frame_processed = cv2.resize(frame.copy(), lanefinder._output_shape)


        # Write the frame to the current video segment
        writer.write(orig_frame)
       
        ret2, buf = cv2.imencode('.jpg', frame_processed)
        if not ret2:
            continue
        jpg = buf.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        )
    if writer is not None:
        writer.release()

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
    return serve_file_from_folder(results_folder, filename, mimetype="video/mp4")

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
        # Build expected MP4 path:
        mp4_name = f"video_{idx_str}.mp4"
        mp4_full = os.path.join(results_folder, mp4_name)
        # Only list it if the .mp4 file actually exists on disk:
        if not os.path.isfile(mp4_full):
            continue

        entries.append({
            "id": idx_str,
            "thumbnail_url": f"{base}/thumbnails/img_{idx_str}.jpg",
            "video_url":     f"{base}/videos/{mp4_name}"
        })

    entries.sort(key=lambda e: int(e["id"]))
    return Response(json.dumps(entries), mimetype='application/json')


@socketio.on('toggleLanefinder')
def on_toggle(data):
    """
    Receives: { "enabled": true|false }
    """
    global lanefinder_enabled
    lanefinder_enabled = bool(data.get('enabled', False))

@socketio.on('updateCameraBias')
def handle_bias(data):
    global camera_offset_angle_bias
    camera_offset_angle_bias = data.get('bias', 0.0)
    print(f"[UPDATE] Camera bias angle: {camera_offset_angle_bias:.2f} degrees")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
