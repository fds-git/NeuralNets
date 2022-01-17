import cv2 
import pickle
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_recognizer = MyFaceRecognizer(device=device, model_path='./mobile_emb128_0.26test_margin2_122000own')

video_session = cv2.VideoCapture(0)
video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

faces = []

name = input("Enter name: ")
id_ = input("Enter id: ")

try:
	database = pd.read_pickle('database.pkl')
except:
	database = pd.DataFrame(columns=['id', 'name', 'centroid', 'centroid_distances'])
	database = database.astype({'id': 'int32', 'name': 'object', 'centroid': 'object', 'centroid_distances': 'object'})


for i in range(5):
	key = cv2. waitKey(1)

	while True:
	     
		ret, frame = video_session.read()
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		boxes, probs = face_recognizer.detector.detect(rgb_frame, landmarks=False)
		boxes, probs = face_recognizer.box_filter(boxes, probs, PROB_TREASHOLD)
		for box in boxes:
			left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
			cv2.rectangle(frame, (left, top), (right, bottom), BLUE_COLOR, LINE_THICKNESS)
		cv2.imshow("Capturing", frame)
		key = cv2.waitKey(10)
		
		if key == ord('s') : 
			if boxes is not None:
				if len(boxes) == 1:
					# При запоминании нового человека в кадре должен быть только один человек
					box = boxes[0]
					face_tensor = face_recognizer.preprocessing(rgb_frame, box)
					faces.append(face_tensor)
					cv2.waitKey(1)
					break


		elif key == ord('q'):
			print("Turning off camera.")
			video_session.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break

video_session.release()
cv2.destroyAllWindows()

faces_batch = torch.cat(faces, dim=0).to(device)
faces_embeddings = face_recognizer.recognizer(faces_batch).detach()
centroid = faces_embeddings.mean(dim=0)
# centroid_distances - даст нам информацию, насколько сильно разнесены представления изображения одного и того же человека в пространстве относительно центроиды
centroid_distances = torch.linalg.vector_norm((faces_embeddings - centroid), ord=2, dim=1)
database = database.append({'id' : id_, 'name' : name, 'centroid': centroid.cpu().numpy(),  'centroid_distances': centroid_distances.cpu().numpy()} , ignore_index=True)
database.to_pickle('database.pkl')