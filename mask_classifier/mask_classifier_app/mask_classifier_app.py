from facenet_pytorch import MTCNN
from mask_classifier_model import MaskClassifier
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
font = cv2.FONT_HERSHEY_SIMPLEX

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mask_classifier = MaskClassifier(device=device, model_path='mask_classifier.pth')
video_session = cv2.VideoCapture(0)

video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

result_mapper = {0: "OK", 1: "Not mask", 2: "Incorrect mask"}


while True  :
	ret, frame = video_session.read()
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	boxes, probs = mask_classifier.face_detector.detect(rgb_frame, landmarks=False)
	boxes, probs = mask_classifier.box_filter(boxes, probs, PROB_TREASHOLD)
	if boxes != []:
		for box, prob in zip(boxes, probs):
			try:
				left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
				cv2.rectangle(frame, (left, top), (right, bottom), BLUE_COLOR, LINE_THICKNESS)

				face_tensor = mask_classifier.preprocessing(rgb_frame, box)
				classes = mask_classifier.mask_classifier(face_tensor)
				max_class = mask_classifier.postprocessing(classes)

				image = cv2.putText(frame, result_mapper[max_class], (left, bottom), font, FONT_SCALE, BLUE_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


			except:
				cv2.imshow('Video', frame)
				continue
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break