"""import cv2
from get_background import get_background
frames = []
MAX_FRAMES = 1000
N = 2
THRESH = 60
ASSIGN_VALUE = 255  # Value to assign the pixel if the threshold is met

cap = cv2.VideoCapture("video.mp4")  # Capture using Computer's Webcam

for t in range(MAX_FRAMES):
    # Capture frame by frame
    ret, frame = cap.read()
    # Convert frame to grayscale

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Append to list of frames
    frames.append(frame_gray)
    if t >= N:
        # D(N) = || I(t) - I(t+N) || = || I(t-N) - I(t) ||
        diff = cv2.absdiff(frames[t - N], frames[t])
        # Mask Thresholding
        threshold_method = cv2.THRESH_BINARY
        ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, threshold_method)
        # Display the Motion Mask
        cv2.imshow('Motion Mask', motion_mask)
"""
"""import cv2
import numpy as np
import time

cap = cv2.VideoCapture("video.mp4")

# time for camera to read at first
time.sleep(2)

_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

_, second_frame = cap.read()
second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

Tx = 50  # threshold value

while (True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    delta_frame1 = cv2.absdiff(gray, first_gray)
    delta_frame2 = cv2.absdiff(gray, second_gray)

    _, thresh_frame1 = cv2.threshold(delta_frame1, Tx, 255, cv2.THRESH_BINARY)
    _, thresh_frame2 = cv2.threshold(delta_frame2, Tx, 255, cv2.THRESH_BINARY)

    # aggregation
    thresh_frame = cv2.bitwise_and(thresh_frame1, thresh_frame2)

    # enhancing
    dilute_frame = cv2.dilate(thresh_frame, None, iterations=3)

    # display
    cv2.imshow("Motion", dilute_frame)
    cv2.imshow("Original", frame)

    # passing frames for next iteration
    second_gray = first_gray
    first_gray = gray

    if (cv2.waitKey(2) == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()"""
"""import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
text = ""

color = (0, 0, 255)  # BGR

cap = cv2.VideoCapture("video.mp4")  # 0 means first webcam

frames = []
counter = 0

threshold = 1

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts captured frame to Grayscale for easier analysis

    frames.append(gray)
    cv2.putText(frame, text, (5, 30), font, 1, color, 3, cv2.LINE_AA)  # may need to change some arguments

    cv2.imshow('frame', frame)

    if counter > 0:
        difference = cv2.subtract(cv2.medianBlur(frames[counter], 15), cv2.medianBlur(frames[counter - 1],
                                                                                      15))  # applies median blur before subtracting for noise reduction
        # cv2.imshow('difference',difference)
        mean = np.mean(difference)
        # print(mean)
        if mean > threshold:
            text = "Motion Detected"
        else:
            text = ""

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit video capture
        break

    counter = counter + 1

cap.release()
cv2.destroyAllWindows()
"""

"""import cv2

cap = cv2.VideoCapture("video_1.mp4")
ret, current_frame = cap.read()
previous_frame = current_frame

current_frame_gray = cap.read()
previous_frame_gray = cap.read()

while(cap.isOpened()):
    #current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)

    cv2.imshow('frame diff ',frame_diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cap.release()
cv2.destroyAllWindows()"""

"""import cv2

videom = cv2.VideoCapture('video.mp4')

ret, frame1 = videom.read()
ret, frame2 = videom.read()

while (videom.isOpened()):
    fark = cv2.absdiff(frame1, frame2)
    gri = cv2.cvtColor(fark, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gri, (5, 5), 0)
    _, esik = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    genis = cv2.dilate(esik, None, iterations=3)
    kontur, _ = cv2.findContours(genis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for k in kontur:
        (x, y, w, h) = cv2.boundingRect(k)
        if cv2.contourArea(k) > 700:
            cv2.rectangle(frame1, (x, y), (w + x, h + y), (0, 0, 255), 2)

    cv2.imshow('feed', frame1)
    frame1 = frame2
    ret, frame2 = videom.read()

    # Eğer q tuşuna basıldı ise oynatmayı durdur.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videom.release()
cv2.destroyAllWindows()"""

"""import numpy as np
import cv2

# Capture video from file
cap=cv2.VideoCapture('video.mp4')

old_frame = None

while True:

    ret, frame = cap.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if old_frame is not None:
            diff_frame = gray - old_frame
            diff_frame -= diff_frame.min()
            disp_frame = np.uint8(255.0*diff_frame/float(diff_frame.max()))
            cv2.imshow('diff_frame',disp_frame)
        old_frame = gray

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    else:
        print('ERROR!')
        break

cap.release()
cv2.destroyAllWindows()"""