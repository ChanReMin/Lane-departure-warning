#!/usr/bin/env python3
"""
ZeroMQ “lane-finder” micro-service.
Listens on ipc:///tmp/lanefinder.ipc
Receives: raw BGR bytes (H×W×3) + 4-byte little-endian width + 4-byte height
Returns : JPEG-encoded inference result
"""
import struct, cv2, zmq, numpy as np, yaml, time
from inference import Lanefinder
from pathlib import Path

# ---------- Utilities ---------------------------------------------------------
def read_config(path = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def as_numpy(msg: bytes) -> np.ndarray:
    """msg == width(4) | height(4) | raw BGR bytes"""
    w, h = struct.unpack("<II", msg[:8])
    img = np.frombuffer(msg[8:], dtype=np.uint8).reshape(h, w, 3)
    return img

def pack_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""

# ---------- Initialise model --------------------------------------------------
cfg = read_config()
lanefinder = Lanefinder(
    model_path   = cfg["model_path"],
    input_shape  = cfg["input_shape"],
    output_shape = tuple(cfg["output_shape"]),
    quant        = cfg["quantization"],
    dequant      = cfg["dequantization"]
)
print("[Python] Lanefinder ready")

# ---------- ZeroMQ loop -------------------------------------------------------
ctx  = zmq.Context()
sock = zmq.Socket(ctx, zmq.REP)
sock.bind("ipc:///tmp/lanefinder.ipc")

while True:
    raw = sock.recv()                     # blocks until C++ sends a frame
    frame = as_numpy(raw)                 # zero-copy view
    frmcpy = frame.copy()                 # keeps original for drawing

    # --- inference ------------------------------------------------------------
    pil_img = lanefinder._preprocess(frame)
    interp  = lanefinder._interpreter

    if interp:
        from pycoral.adapters.common import set_input
        set_input(interp, pil_img)
        t0 = time.perf_counter()
        interp.invoke()
        output = interp.get_tensor(interp.get_output_details()[0]['index'])
        result = lanefinder._postprocess(output, frmcpy)
        print(f"[Python] Infer {1e3*(time.perf_counter()-t0):.1f} ms")
    else:                                 # no TPU detected
        result = cv2.putText(
            cv2.resize(frmcpy, lanefinder._output_shape),
            "TPU NOT DETECTED!", (40,60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2
        )

    sock.send(pack_jpeg(result))          # JPEG bytes back to C++
