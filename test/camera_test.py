#coding:utf-8
import sys
import argparse
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np


if len(sys.argv) !=3:
    print("Usage: python camera_test.py <video source> <scale size>")
    print(" scale should be 0.1 to 1")
    sys.exit()

videopath = sys.argv[1]

try:
    scale = float(sys.argv[2])
except:
    print("Scale size is not float.")
    sys.exit()

if scale > 1 or scale < 0.1:
    print("Scale is out of range, use 0.5 as default")
    scale = 0.5

test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)


video_capture = cv2.VideoCapture(videopath)
corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()

    if ret:
        height, width, channels = frame.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        frame_resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image

        image = np.array(frame_resized)
        boxes_c,landmarks = mtcnn_detector.detect(image)

        print landmarks.shape
        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]/scale), int(bbox[1]/scale), int(bbox[2]/scale), int(bbox[3]/scale)]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])/2):
                cv2.circle(frame, (int(landmarks[i][2*j]/scale),int(int(landmarks[i][2*j+1])/scale)), 2, (0,0,255))
        # time end
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print 'device not find'
        break

video_capture.release()
cv2.destroyAllWindows()
