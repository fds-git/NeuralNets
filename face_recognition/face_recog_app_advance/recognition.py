from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import cv2
import numpy as np
import torch
import pandas as pd
from model import MyFaceRecognizer

FPS = 10
WIDTH = 640
HEIGHT = 480
PROB_TREASHOLD = 0.9
FONT_SCALE = 1.0
TEXT_THICKNESS = 1
LINE_THICKNESS = 2
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)
DISTANCE_TREASHOLD = 1.0 # Пороговое расстояние между эмбэддингами лиц

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_recognizer = MyFaceRecognizer(device=device, database_path='database.pkl')
video_session = cv2.VideoCapture(0)

video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


while True  :
	ret, frame = video_session.read()
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	boxes, probs = face_recognizer.detector.detect(rgb_frame, landmarks=False)
	boxes, probs = face_recognizer.box_filter(boxes, probs, PROB_TREASHOLD)
	if boxes != []:
		for box, prob in zip(boxes, probs):
			try:
				face_tensor = face_recognizer.preprocessing(rgb_frame, box)
				face_embedding = face_recognizer.recognizer(face_tensor).detach()
				nearest_name = face_recognizer.get_nearest_name(face_embedding, DISTANCE_TREASHOLD)

				left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
				cv2.rectangle(frame, (left, top), (right, bottom), BLUE_COLOR, LINE_THICKNESS)
				cv2.putText(frame, nearest_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE_COLOR, TEXT_THICKNESS)
			except:
				cv2.imshow('Video', frame)
				continue
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break