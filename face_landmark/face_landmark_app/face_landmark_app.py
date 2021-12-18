from facenet_pytorch import MTCNN
from face_landmark_model import LandmarkDetector
import cv2
import numpy as np
import torch
import pandas as pd

FPS = 10
WIDTH = 640
HEIGHT = 480
PROB_TREASHOLD = 0.9
FONT_SCALE = 1.0
TEXT_THICKNESS = 1
LINE_THICKNESS = 2
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
landmark_detector = LandmarkDetector(device=device, model_path='landmark_detection_xception_50ep_128_lr0.003_gamma0.95')
video_session = cv2.VideoCapture(0)

video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


while True  :
	ret, frame = video_session.read()
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	boxes, probs = landmark_detector.face_detector.detect(rgb_frame, landmarks=False)
	boxes, probs = landmark_detector.box_filter(boxes, probs, PROB_TREASHOLD)
	if boxes != []:
		for box, prob in zip(boxes, probs):
			try:
				left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
				cv2.rectangle(frame, (left, top), (right, bottom), BLUE_COLOR, LINE_THICKNESS)

				face_tensor = landmark_detector.preprocessing(rgb_frame, box)
				landmarks = landmark_detector.landmark_detector(face_tensor)
				landmarks = landmark_detector.postprocessing(landmarks, box)

				for landmark in landmarks:
					cv2.circle(frame, (landmark[0], landmark[1]), 1, (0, 0, 255), -1)

			except:
				cv2.imshow('Video', frame)
				continue
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break