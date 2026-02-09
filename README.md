# 🤖 Video Background Remover (RVM)

A high-performance video background removal tool optimized for **Apple Silicon (M1/M2/M3)**. This tool uses Robust Video Matting (RVM) with a MobileNetV3 backbone to replace backgrounds with a green screen while preserving audio.

## 🚀 Features
* **M1 Optimized:** Uses Apple's Metal Performance Shaders (MPS) for GPU acceleration.
* **Temporal Consistency:** Uses recurrent states (r1-r4) to track subjects accurately over time.
* **Audio Preservation:** Automatically merges the original audio back into the processed video using FFmpeg.
* **Flexible Formats:** Supports .mp4, .mov, and more.

## 🛠 Prerequisites

### 1. System Requirements
* **macOS** (Optimized for MacBook Pro M1)
* **FFmpeg**: Required for audio merging.
  ```bash
  brew install ffmpeg
  ```

### 2. Python Dependencies
Ensure you are in your virtual environment (source venv/bin/activate), then install:
```bash
pip install torch torchvision opencv-python tqdm
```

### 3. Model Weight
Download `rvm_mobilenetv3_fp32.torchscript` and place it in the root directory of this project.

## 💻 Usage

Run the script from your terminal using the following arguments:

### Basic Command
```bash
python remove_bg.py --input input.mov
```

### Custom Output and Quality
```bash
python remove_bg.py --input robot.mp4 --output finished.mp4 --ratio 0.4
```

### Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| --input | *Required* | Path to your source video file. |
| --output | None | Custom output path. Defaults to filename_green.ext. |
| --ratio | 0.5 | Downsample ratio. Use 0.25 for 4K or 0.6 for small subjects. |

## ⚙️ How it Works
The script processes video frames through a recurrent neural network. Unlike standard "background removers" that look at frames individually, this model "remembers" the subject's position, which prevents flickering.

1. **Inference:** Extracts the alpha mask (pha) and foreground (fgr).
2. **Sharpening:** Applies a contrast boost to the mask to ensure white/gray limbs aren't lost against similar backgrounds.
3. **Muxing:** Uses FFmpeg to "zip" the original audio track onto the new green-screen video.

## ⚖️ License
This project uses the Robust Video Matting (RVM) weights. Please refer to their official repository for license details.