import json
import os
import pandas as pd
from datasets import load_dataset

# Load the training split
dataset = load_dataset("HuggingFaceM4/something-something-v2", data_dir='/data/datasets/something-something-v2', split="train")
video_dir = "/data/datasets/something-something-v2/20bn-something-something-v2"  # Adjust to your video directory

# Create a list to store data
data = []

# Iterate through the dataset
for item in dataset:
    video_id = item["video_id"]
    label = item["label"]
    text = item["text"]
    # Construct the video path (adjust based on actual file structure)
    video_path = os.path.join(video_dir, f"{video_id}.webm")
    data.append({
        "path": video_path,
        "video_id": video_id,
        "label": label,
        "text": text
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("ssv2_train_paths.csv", index=False)
print("Saved ssv2_train_paths.csv")