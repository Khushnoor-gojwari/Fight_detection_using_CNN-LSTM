# CNN-LSTM-Violence-detection
This project is a re-implementation of the model described in the paper : https://ieeexplore.ieee.org/document/8852616
# Dependencies
- Python ( I used 3.8.3)
- opencv
- tensorflow 2.x
- keras 
# Quick start (fightdetection.py)
This adds a simple menu-driven script that runs violence detection on a video
file or a live camera, shows the result live in a window, and auto-saves the
annotated video to `output_videos/`.

```bash
# 1. Create and activate a virtual environment (Python 3.12 tested)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model weights
#    - fightw.hdfs (from Google Drive)
#    - VGG19 ImageNet base weights (cached by Keras)
python download_weights.py

# 4. Run detection (interactive menu: 1 = camera, 2 = video file)
python fightdetection.py
```

Non-interactive options:

```bash
python fightdetection.py -i Data/fight.mp4     # detect on a video file
python fightdetection.py -c 0                  # detect on camera index 0
```

Notes:
- The annotated output is saved automatically inside `output_videos/`.
- Press `q` in the window to stop. The first ~1 second shows "Warming up"
  while the 30-frame buffer fills.
- Runs on CPU if no CUDA GPU is present (slower, but works).

# Use 
- make sure you have all the necessary dependencies like Tensorflow 2, Keras, numpy, opencv, especially cuda tools for gpu support as the process is computationally heavy. 
- Clone the project and download the trained weights and put them in the same directory (you can put them wherever you want but then you must add the path for the weights in the "violence_model.py" file).
- Run the detector with the interactive menu, or pass a video path directly:
 ``` 
 python fightdetection.py                              # menu: 1 = camera, 2 = video
 python fightdetection.py -i Data/fight.mp4            # run on a specific video
 ```
- The annotated result is saved automatically inside `output_videos/`. 
# script at work :
The model takes 30 frames as an input. The overlay shows a rolling-average
violence probability while the video plays.

## Trained weights : https://drive.google.com/file/d/1phOcWnglLZxsly7gKBab8XJuKzFf5vdp/view?usp=sharing

## Datasets : 
- Movies Fight Detection Dataset :  https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635
- Hockey Fight Detection Dataset : https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89
- VIOLENT-FLOWS DATABASE  : 
https://www.openu.ac.il/home/hassner/data/violentflows/

## Credits & attribution
- Model architecture and trained weights (`violence_model.py`, `fightw.hdfs`):
  based on the [CNN-LSTM Violence Detection](https://github.com/souhaiel1/CNN-LSTM-Violence-detection)
  project by souhaiel1, a re-implementation of the paper
  [IEEE 8852616](https://ieeexplore.ieee.org/document/8852616).
- My contribution (Khushnoor): `fightdetection.py` (menu-driven live/video
  detection with multi-format input and auto-saved output), `download_weights.py`,
  and `requirements.txt`.
