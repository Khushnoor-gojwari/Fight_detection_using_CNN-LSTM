"""
Fight / Violence detection.

Menu-driven terminal script:
    1 -> Realtime detection from a webcam / connected camera.
    2 -> Detection from an input video file (you provide the path).

The model is the CNN(VGG19)+LSTM architecture from violence_model.py and the
pre-trained weights in "fightw.hdfs".

Every run is displayed live in a window (like YOLOv8) AND the annotated
result is automatically saved into the "output_videos" folder.

Usage:
    python fightdetection.py

Press 'q' in the display window to quit early.
"""

from __future__ import absolute_import, division, print_function

import os

# The pre-trained weights were saved with the legacy (Keras 2) API.
# On Python 3.12 / TensorFlow >= 2.16 the default is Keras 3, which is not
# compatible with the old model definition (e.g. optimizer `lr=` argument and
# the HDF5 weight layout). Force TF to use the Keras 2 backend (tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Quieten TensorFlow's very chatty startup logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from violence_model import khushnoor_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIME_STEPS = 30            # number of frames the model looks at per prediction
FRAME_H = 160              # frame height expected by the model
FRAME_W = 160              # frame width expected by the model
WEIGHTS = "fightw.hdfs"    # pre-trained weights (already in this directory)
OUTPUT_DIR = "output_videos"

# On CPU the model is heavy, so we do NOT predict on every single frame.
# We keep the video playing smoothly and run a prediction every N frames.
PREDICT_EVERY = 5

# Decision threshold for calling a clip "violence".
# The model's output layer is a sigmoid, so the violence score naturally sits
# high (often 0.5-0.9) even for non-violent motion. The original project only
# treated a clip as violent when the score crossed ~0.9, so we use the same.
# Lower this if you want it more sensitive; raise it to reduce false alarms.
VIOLENCE_THRESHOLD = 0.90

NORMAL_LABEL = "Normal"
VIOLENCE_LABEL = "Violence"


def load_model(weights=WEIGHTS):
    """Build the CNN-LSTM architecture and load the pre-trained weights."""
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            "Could not find weights file '{}'. Place the trained weights in "
            "this directory or pass the correct path.".format(weights)
        )
    print("[INFO] building model and loading weights (this can take a moment)...")
    model = khushnoor_model(tf, w=weights)
    print("[INFO] model ready.")
    return model


def preprocess(frame):
    """Resize a BGR frame to the model input and scale to [0, 1]."""
    resized = resize(frame, (FRAME_H, FRAME_W, 3))
    if np.max(resized) > 1:
        resized = resized / 255.0
    return resized


def predict_window(model, window, avg_queue):
    """Run the model on a 30-frame window and return (label, probability).

    `window` is a deque of TIME_STEPS pre-processed frames.
    `avg_queue` keeps recent violence probabilities for rolling averaging.
    """
    datav = np.zeros((1, TIME_STEPS, FRAME_H, FRAME_W, 3), dtype=np.float32)
    datav[0] = np.array(window, dtype=np.float32)

    preds = model.predict(datav, verbose=0)
    violence_prob = float(preds[0][1])

    # Rolling average smooths out per-frame jitter.
    avg_queue.append(violence_prob)
    smoothed = float(np.mean(avg_queue))

    # Label is decided by the threshold; the returned probability is always the
    # (smoothed) violence probability so the overlay stays consistent.
    label = VIOLENCE_LABEL if smoothed >= VIOLENCE_THRESHOLD else NORMAL_LABEL
    return label, smoothed


def draw_overlay(frame, label, prob, violence_prob, fps, warming_up):
    """Draw a YOLO-style overlay: status banner, confidence bar and FPS."""
    h, w = frame.shape[:2]
    is_violence = label == VIOLENCE_LABEL
    color = (0, 0, 255) if is_violence else (0, 200, 0)

    # Semi-transparent banner across the top.
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if warming_up:
        status = "Warming up... buffering frames"
        status_color = (0, 215, 255)
    else:
        status = "{}  |  violence {:.0f}%".format(label, violence_prob * 100)
        status_color = color

    cv2.putText(frame, status, (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, status_color, 2, cv2.LINE_AA)

    # FPS in the top-right corner.
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Violence-probability bar along the bottom.
    bar_x, bar_y, bar_w, bar_h = 15, h - 35, w - 30, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), 1)
    fill = int(bar_w * max(0.0, min(1.0, violence_prob)))
    bar_color = (0, 0, 255) if violence_prob >= VIOLENCE_THRESHOLD else (0, 200, 0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  bar_color, -1)
    cv2.putText(frame, "violence prob {:.0f}%".format(violence_prob * 100),
                (bar_x + 5, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Red border flash when violence is detected.
    if is_violence and not warming_up:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

    return frame


def build_output_path(source):
    """Return an auto output path inside OUTPUT_DIR based on the source."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if isinstance(source, int):
        name = "camera{}_{}".format(source, datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        base = os.path.splitext(os.path.basename(source))[0]
        name = "{}_result".format(base)
    return os.path.join(OUTPUT_DIR, name + ".avi")


class FrameReader:
    """Unified frame reader for cameras, .mp4, .avi and .gif inputs.

    Uses OpenCV for everything. If the input is a GIF that OpenCV cannot
    decode (depends on the build), it transparently falls back to imageio.
    Always yields 3-channel BGR frames via read() -> (grabbed, frame).
    """

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.use_imageio = False
        self.fps = 20.0
        self._frames = None
        self._idx = 0
        self._open()

    def _open(self):
        is_gif = isinstance(self.source, str) and \
            self.source.lower().endswith(".gif")

        self.cap = cv2.VideoCapture(self.source)
        opened = self.cap.isOpened()

        if opened:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self.fps = fps

        # For GIFs, make sure OpenCV can really decode a frame; if not, fall
        # back to imageio (robust across OpenCV builds).
        if is_gif:
            if opened and self.cap.grab():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
                self._open_imageio()

    def _open_imageio(self):
        import imageio.v3 as iio
        self.use_imageio = True
        self._frames = iio.imread(self.source)  # shape (n, H, W, C)
        if self._frames.ndim == 3:  # a single still image
            self._frames = self._frames[None, ...]
        self._idx = 0
        # Try to derive fps from the GIF frame duration metadata.
        try:
            meta = iio.immeta(self.source)
            duration = meta.get("duration")  # seconds per frame
            if duration:
                self.fps = max(1.0, 1.0 / float(duration))
            else:
                self.fps = 10.0
        except Exception:
            self.fps = 10.0

    def isOpened(self):
        if self.use_imageio:
            return self._frames is not None and len(self._frames) > 0
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.use_imageio:
            if self._idx >= len(self._frames):
                return False, None
            frame = np.asarray(self._frames[self._idx])
            self._idx += 1
            if frame.ndim == 2:  # grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


def run(source, output_path=None, avg_size=128, predict_every=PREDICT_EVERY):
    """Run detection on a video source and always save the annotated output.

    source: an int (camera index) or a path to a video file.
    output_path: where to save the annotated video (auto-generated if None).
    """
    model = load_model()

    cap = FrameReader(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source: {}".format(source))
        return

    if output_path is None:
        output_path = build_output_path(source)

    # Use the source FPS for files/gifs; a sensible default for live cameras.
    out_fps = cap.fps if (cap.fps and cap.fps > 1) else 20.0

    writer = None
    (W, H) = (None, None)

    window = deque(maxlen=TIME_STEPS)   # rolling frame buffer
    avg_queue = deque(maxlen=avg_size)  # rolling probability buffer
    label, prob, violence_prob = NORMAL_LABEL, 1.0, 0.0

    frame_count = 0
    fps = 0.0
    last_t = time.time()
    can_display = True  # disabled automatically if no GUI is available

    print("[INFO] running... a window will open. Press 'q' to quit.")
    print("[INFO] saving annotated video to: {}".format(output_path))

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (W, H), True)

        frame_count += 1
        window.append(preprocess(frame))
        warming_up = len(window) < TIME_STEPS

        # Predict periodically (heavy on CPU) once the buffer is full.
        if not warming_up and frame_count % predict_every == 0:
            label, prob = predict_window(model, window, avg_queue)
            violence_prob = float(np.mean(avg_queue))

        # Rough FPS estimate for the on-screen readout.
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
        last_t = now

        output = draw_overlay(frame.copy(), label, prob, violence_prob,
                              fps, warming_up)

        if writer is not None:
            writer.write(output)

        if can_display:
            try:
                cv2.imshow("Fight Detection", output)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            except cv2.error:
                # No GUI backend / no display available: keep saving silently.
                can_display = False
                print("[WARN] no display available; continuing without a live "
                      "window (video is still being saved).")

    print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] done. Output saved to: {}".format(output_path))


def menu():
    """Interactive terminal menu."""
    print("=" * 50)
    print(" Fight / Violence Detection")
    print("=" * 50)
    print(" 1 - Realtime detection (webcam / camera)")
    print(" 2 - Detection from a video file")
    print("=" * 50)

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        cam = input("Camera index [default 0]: ").strip()
        cam_index = int(cam) if cam else 0
        run(cam_index)
    elif choice == "2":
        path = input("Enter the path to the input video: ").strip()
        # Strip surrounding quotes if the user pasted a quoted path.
        path = path.strip('"').strip("'")
        if not os.path.isfile(path):
            print("[ERROR] File not found: {}".format(path))
            return
        run(path)
    else:
        print("[ERROR] Invalid choice. Please run again and pick 1 or 2.")


def main():
    parser = argparse.ArgumentParser(
        description="Fight / violence detection from video files or a live camera."
    )
    parser.add_argument(
        "-i", "--input",
        help="Path to an input video file. If omitted, an interactive menu is shown.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the annotated output video "
             "(default: auto-named inside output_videos/).",
    )
    parser.add_argument(
        "-c", "--camera", type=int,
        help="Camera index for realtime detection (e.g. 0).",
    )
    parser.add_argument(
        "-s", "--size", type=int, default=128,
        help="Size of the rolling-average window (default 128).",
    )
    args = parser.parse_args()

    # Command-line mode (skips the menu) if flags are provided.
    if args.camera is not None:
        run(args.camera, output_path=args.output, avg_size=args.size)
    elif args.input:
        run(args.input, output_path=args.output, avg_size=args.size)
    else:
        menu()


if __name__ == "__main__":
    main()
