import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import subprocess
import os
import argparse

# --- 1. COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="AI Background Remover for Mac M1")
parser.add_argument("--input", type=str, required=True, help="Path to input video")
parser.add_argument("--output", type=str, help="Path to output video")
parser.add_argument("--ratio", type=float, help="Downsample ratio (0.1 to 1.0). Auto-calculated if omitted.")
parser.add_argument("--quality", type=int, default=8, help="Quality scale 1 (Smallest) to 10 (Best). Default is 8.")
args = parser.parse_args()

input_path = args.input

# --- 2. VIDEO CAPTURE SETUP ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Could not open {input_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 3. AUTO-RATIO & MAPPING QUALITY ---
if args.ratio is None:
    auto_ratio = 512 / min(width, height)
    actual_ratio = max(min(auto_ratio, 1.0), 0.1)
    print(f"Auto-calculated ratio: {actual_ratio:.3f}")
else:
    actual_ratio = args.ratio

# Map intuitive 1-10 scale to FFmpeg CRF (32 to 17)
crf_val = int(35 - (args.quality * 1.8))
crf_val = max(min(crf_val, 32), 17)

if not args.output:
    base, ext = os.path.splitext(input_path)
    final_output = f"{base}_green{ext}"
else:
    final_output = args.output

temp_output = "temp_no_audio.mp4"

# --- 4. MODEL SETUP ---
model = torch.jit.load("rvm_mobilenetv3_fp32.torchscript")
model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
transform = transforms.ToTensor()

r1, r2, r3, r4 = None, None, None, None

print(f"Processing on {device} (Quality Scale: {args.quality}/10)...")

# --- 5. PROCESSING LOOP ---
with torch.no_grad():
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device).to(torch.float32)

        # Inference
        fgr, pha, r1, r2, r3, r4 = model(tensor, r1, r2, r3, r4, float(actual_ratio))

        mask = pha[0][0].cpu().numpy()
        mask = np.where(mask > 0.15, mask * 1.3, mask * 0.5)
        mask = np.clip(mask, 0, 1)
        mask = np.expand_dims(mask, axis=2)

        green_bg = np.zeros_like(frame)
        green_bg[:] = (0, 255, 0)

        output = (frame * mask + green_bg * (1 - mask)).astype(np.uint8)
        out.write(output)

cap.release()
out.release()

# --- 6. AUDIO MERGE & COMPRESSION (FFmpeg) ---
if os.path.exists(temp_output):
    print(f"\nMerging audio and compressing (Target CRF: {crf_val})...")
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_output,
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'libx264',
        '-crf', str(crf_val),
        '-preset', 'slow',
        '-c:a', 'aac', '-b:a', '128k',
        final_output
    ]
    subprocess.run(cmd)
    if os.path.exists(final_output):
        os.remove(temp_output)
    print(f"\nSuccess! File ready: {final_output}")
