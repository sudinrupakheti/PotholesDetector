import os
import kagglehub

os.system("pip install -q ultralytics")

# Download dataset
path = kagglehub.dataset_download("sudinrupakheti/potholes")

# Train (use existing data.yaml in dataset)
os.system(f"yolo detect train data={path}/data.yaml model=yolov8n.pt epochs=50 device=0")

# Copy weights
import shutil
for root, dirs, files in os.walk('/kaggle/working/runs'):
    if 'best.pt' in files:
        src = os.path.join(root, 'best.pt')
        shutil.copy(src, '/kaggle/working/best.pt')
        print(f"✓ Saved: {src}")
        break
