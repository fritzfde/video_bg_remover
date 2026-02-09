import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import subprocess
import os
import argparse

# --- COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="AI Background Remover for Mac M1")
parser.add_argument("--input", type=str, required=True, help="Path to input video")
parser.add_argument("--output", type=str, help="Path to output video (with extension)")
parser.add_argument("--ratio", type=float, default=0.5, help="Downsample ratio (0.1 to 1.0)")
args = parser.parse_args()

input_path = args.input

# If no output name is provided, generate one based on the input
if not args.output:
    base, ext = os.path.splitext(input_path)
    final_output = f"{base}_green{ext}"
else:
    final_output = args.output

# Determine the extension for the temp file (FFmpeg needs an intermediate container)
output_ext = os.path.splitext(final_output)[1].lower()
temp_output = f"temp_no_audio{output_ext}"

# --- MODEL SETUP ---
model = torch.jit.load("rvm_mobilenetv3_fp32.torchscript")
model.eval()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Could not open {input_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use different codecs based on extension
if output_ext == ".mov":
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # High quality for MOV
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
transform = transforms.ToTensor()

# States for your specific TorchScript model
r1, r2, r3, r4 = None, None, None, None

print(f"Processing on {device}...")
print(f"Output will be saved as: {final_output}")

with torch.no_grad():
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device).to(torch.float32)

        # Passing states individually
        fgr, pha, r1, r2, r3, r4 = model(tensor, r1, r2, r3, r4, float(args.ratio))

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

# --- AUDIO MERGE & FORMAT CONVERSION ---
if os.path.exists(temp_output):
    print("\nMerging audio and finalizing format...")
    # FFmpeg is smart enough to handle the .mov vs .mp4 containers automatically
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_output,
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'copy', # Copy the video we already processed
        '-c:a', 'aac',  # Good standard audio
        final_output
    ]
    subprocess.run(cmd)
    os.remove(temp_output)
    print(f"\nSuccess! File ready: {final_output}")