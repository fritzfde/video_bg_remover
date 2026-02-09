import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import subprocess
import os

# 1. Setup Model
model = torch.jit.load("rvm_mobilenetv3_fp32.torchscript")
model.eval()

# Optimized for your MacBook Pro M1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

input_path = "input.mp4"
temp_output = "temp_no_audio.mp4"
final_output = "output_with_audio.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Downsample ratio: 0.5 is usually the sweet spot for 1080p
downsample_ratio = torch.tensor([0.5], dtype=torch.float32).to(device)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

transform = transforms.ToTensor()

# 1. Initialize states as four separate None objects
r1, r2, r3, r4 = None, None, None, None

print(f"Processing on {device}...")

with torch.no_grad():
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device).to(torch.float32)

        # 2. Pass them explicitly according to the model's 'Declaration'
        # forward(src, r1, r2, r3, r4, downsample_ratio)
        # Note: we convert downsample_ratio to a float because your model asks for 'float'
        fgr, pha, r1, r2, r3, r4 = model(tensor, r1, r2, r3, r4, float(downsample_ratio))

        # Process the mask
        mask = pha[0][0].cpu().numpy()

        # IMPROVED MASK LOGIC for your white robot:
        # We increase the contrast of the mask to keep the limbs
        mask = np.where(mask > 0.1, mask * 1.2, mask * 0.5)
        mask = np.clip(mask, 0, 1)

        mask = np.expand_dims(mask, axis=2)

        green_bg = np.zeros_like(frame)
        green_bg[:] = (0, 255, 0)

        output = (frame * mask + green_bg * (1 - mask)).astype(np.uint8)
        out.write(output)

cap.release()
out.release()

# --- AUDIO MERGE (Requires brew install ffmpeg) ---
if os.path.exists(temp_output):
    print("Merging audio...")
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_output,
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'copy',
        '-c:a', 'aac',
        final_output
    ]
    subprocess.run(cmd)
    if os.path.exists(final_output):
        os.remove(temp_output)
        print(f"Finished! Final video: {final_output}")
