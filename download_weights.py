"""
Download the pre-trained model weights.

1. fightw.hdfs  -> the trained fight-detection weights, fetched from Google
   Drive (see the link in README.md).
2. VGG19 (ImageNet) base weights -> triggered here so teammates don't have to
   wait for the download on their first real run. Keras caches these under
   ~/.keras/models, so this only downloads once.

Usage:
    python download_weights.py
"""

import os

# Google Drive file id for fightw.hdfs (from the README share link):
# https://drive.google.com/file/d/1phOcWnglLZxsly7gKBab8XJuKzFf5vdp/view
GDRIVE_FILE_ID = "1phOcWnglLZxsly7gKBab8XJuKzFf5vdp"
WEIGHTS_FILE = "fightw.hdfs"


def download_fight_weights():
    """Download fightw.hdfs from Google Drive if it is not already present."""
    if os.path.isfile(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 0:
        print("[OK] '{}' already exists, skipping download.".format(WEIGHTS_FILE))
        return

    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "gdown is not installed. Run 'pip install -r requirements.txt' "
            "first (or 'pip install gdown')."
        )

    url = "https://drive.google.com/uc?id={}".format(GDRIVE_FILE_ID)
    print("[INFO] downloading {} from Google Drive...".format(WEIGHTS_FILE))
    gdown.download(url, WEIGHTS_FILE, quiet=False)

    if not os.path.isfile(WEIGHTS_FILE) or os.path.getsize(WEIGHTS_FILE) == 0:
        raise SystemExit(
            "Download failed. Please download the weights manually from the "
            "Google Drive link in README.md and place '{}' in this folder."
            .format(WEIGHTS_FILE)
        )
    print("[OK] saved {} ({} bytes).".format(
        WEIGHTS_FILE, os.path.getsize(WEIGHTS_FILE)))


def download_vgg19_weights():
    """Pre-fetch and cache the VGG19 ImageNet base weights via Keras."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print("[INFO] fetching VGG19 (ImageNet) base weights (cached by Keras)...")
    import tensorflow as tf
    tf.keras.applications.vgg19.VGG19(
        include_top=False, weights="imagenet", input_shape=(160, 160, 3)
    )
    print("[OK] VGG19 base weights are ready.")


if __name__ == "__main__":
    download_fight_weights()
    download_vgg19_weights()
    print("\nAll weights are ready. You can now run: python fightdetection.py")
