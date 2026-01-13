import os
import cv2
import csv
import mediapipe as mp

# Correct path – alphabet PNG folders are inside dataset/asl_alphabet_train
ASL_PATH = "dataset/asl_alphabet_train"

# Output folder for CSV
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1)

print("\n🔄 Converting alphabet PNG → CSV...\n")

for label in os.listdir(ASL_PATH):
    folder_path = os.path.join(ASL_PATH, label)

    # Only process folders named A, B, C, ..., Z
    if not os.path.isdir(folder_path):
        continue
    if len(label) != 1 or not label.isalpha():
        continue

    csv_file = os.path.join(OUT_DIR, f"{label}.csv")
    print(f"📌 Processing: {label}")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                row = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)

print("\n🎉 DONE! Alphabet CSV files created inside dataset/.")
print("➡ Now run: python train_model.py")
