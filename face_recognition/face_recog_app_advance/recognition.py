from facenet_pytorch import MTCNN
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import torch

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
mtcnn = MTCNN(keep_all=True, device='cuda:0')		
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()			
video_session = cv2.VideoCapture(0)

video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


while True  :
	# Grab a single frame of video
	ret, frame = video_session.read()
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


	#boxes, probs = mtcnn.detect(rgb_small_frame, landmarks=False)
	boxes, probs = mtcnn.detect(rgb_frame, landmarks=False)
	if boxes is not None:
		for box, prob in zip(boxes, probs):
			if prob >= PROB_TREASHOLD:
				left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
				starting_point = (left, top)
				ending_point = (right, bottom)

				cv2.rectangle(frame, starting_point, ending_point, BLUE_COLOR, LINE_THICKNESS)
				cv2.putText(frame, str(f'{prob:.2f}'), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE_COLOR, TEXT_THICKNESS)
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		# t.cancel()
		break



#norm = (faces_embeddings - centroid).norm(dim=1).detach()