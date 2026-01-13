import cv2
import mediapipe as mp
import csv
import os

# Output folder
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Ask user which word to record
label = input("Enter WORD label (HELLO / HELP / YES / NO / STOP / THANKYOU): ").upper()

save_path = os.path.join(DATA_DIR, f"{label}.csv")

cap = cv2.VideoCapture(0)
print(f"📸 Recording data for WORD = {label}")
print("👉 Move your hand naturally and repeat the gesture.")
print("👉 Press 'q' to stop.")

with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                # Save 21 landmark values
                row = []
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                writer.writerow(row)

                # Draw landmarks on screen
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cv2.putText(img, f"Recording: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Record Word Dataset", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
