import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Load model
model = torch.jit.load("rvm_mobilenetv3_fp32.torchscript")
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Video input/output
input_path = "input.mp4"
output_path = "output_green.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

transform = transforms.ToTensor()

# Initialize recurrent states as None for the first frame
rec_states = None

with torch.no_grad():
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        # Pass rec_states and update them with the model's output
        # fgr = foreground, pha = alpha mask, rec_states = temporal memory
        fgr, pha, *rec_states = model(tensor, *rec_states) if rec_states is not None else model(tensor)

        mask = pha[0][0].cpu().numpy()

        # --- IMPROVEMENT: MASK SHARPENING ---
        # Any pixel > 0.3 probability becomes more solid (helps with limb detection)
        mask = np.clip(mask * 1.5, 0, 1)

        green_bg = np.zeros_like(frame)
        green_bg[:] = (0, 255, 0)

        mask = np.expand_dims(mask, axis=2)

        # Combine using the improved mask
        output = frame * mask + green_bg * (1 - mask)
        out.write(output.astype(np.uint8))

cap.release()
out.release()
print("Finished.")

