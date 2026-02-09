# 🤖 Video Background Remover (RVM)

A high-performance video background removal tool optimized for **Apple Silicon (M1/M2/M3)**.

## 🚀 Features
* **M1 Optimized:** Uses Apple's Metal Performance Shaders (MPS) for GPU acceleration.
* **Auto-Ratio:** Automatically adjusts internal resolution based on video size.
* **Intuitive Quality:** Simple 1-10 compression scale.
* **Audio Preservation:** Merges original audio back via FFmpeg.

## 🛠 Prerequisites

### 1. System Requirements
* **macOS** (Optimized for MacBook Pro M1)
* **FFmpeg**: Required for compression and audio.
  ```bash
  brew install ffmpeg
  ```

### 2. Python Dependencies
```bash
pip install torch torchvision opencv-python tqdm
```

### 3. Model Weight
Download `rvm_mobilenetv3_fp32.torchscript` and place it in the root directory.

## 💻 Usage

### Basic Command
```bash
python remove_bg.py --input input.mov
```

### High Quality Render
```bash
python remove_bg.py --input robot.mp4 --quality 10 --ratio 0.7
```

### Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| --input | *Req* | Path to source video. |
| --output | None | Custom output path. |
| --ratio | Auto | Scale (0.1-1.0). Higher = more detail. |
| --quality | 8 | 1 (Smallest file) to 10 (Best quality). |

## ⚙️ How it Works
1. **Inference:** Extracts alpha mask and foreground using RVM.
2. **Sharpening:** Boosts mask contrast for better edges on robots.
3. **Encoding:** Uses FFmpeg with libx264 and CRF mapping for efficient file sizes.

## ⚖️ License
Refer to the official Robust Video Matting (RVM) repository for license details.