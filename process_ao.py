import numpy as np
from thetan.berserker import berserker2

import cv2
from tqdm import tqdm

# Path to the video file
# video_path = r'/Users/prajwal/Desktop/P2Video/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'

video_path = r'/home/elef/Downloads/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'


cap = cv2.VideoCapture(video_path)

frames = []
while True:
    # Read frame from the video
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Finished reading the video. Exiting ...")
        break

    # Append frame to the list
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames.append(gray_frame)

cap.release()

frames = np.asarray(frames)
frames = np.swapaxes(frames, 0, 2)
frames = np.swapaxes(frames, 0, 1)

print(frames.shape)

t = 1