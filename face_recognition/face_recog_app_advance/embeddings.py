import cv2 
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

FPS = 10
WIDTH = 640
HEIGHT = 480

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=True, device=device)
recognizer = InceptionResnetV1(pretrained='vggface2', device=device).eval()
video_session = cv2.VideoCapture(0)
video_session.set(cv2.CAP_PROP_FPS, FPS)
video_session.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

faces = []

name=input("enter name: ")
ref_id=input("enter id: ")

try:
	f=open("ref_name.pkl","rb")

	ref_dictt=pickle.load(f)
	f.close()
except:
	ref_dictt={}
ref_dictt[ref_id]=name


f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()

try:
	f=open("ref_embed.pkl","rb")

	embed_dictt=pickle.load(f)
	f.close()
except:
	embed_dictt={}

for i in range(4):
	key = cv2. waitKey(1)

	while True:
	     
		ret, frame = video_session.read()
		cv2.imshow("Capturing", frame) # frame - 640x480
		key = cv2.waitKey(10)
		
		if key == ord('s') : 
			#face_locations = face_recognition.face_locations(rgb_small_frame)
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			boxes, probs = detector.detect(rgb_frame, landmarks=False)
			#print(face_locations)
			if boxes is not None:
				box = boxes[0]
				left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
				#print(rgb_frame.shape)
				face_numpy = frame[top:bottom, left:right, :]
				face_numpy = cv2.resize(face_numpy, (160, 160))
				face_tensor = torch.from_numpy(face_numpy)
				face_tensor = face_tensor.permute(2,0,1)
				face_tensor = torch.unsqueeze(face_tensor, 0)
				face_tensor = face_tensor.float()
				faces.append(face_tensor)
				cv2.waitKey(1)
				#cv2.destroyAllWindows()     
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
faces_embeddings = recognizer(faces_batch).detach()
centroid = faces_embeddings.mean(dim=0).cpu().numpy()

embed_dictt[ref_id]=[centroid]

f=open("ref_embed.pkl","wb")
pickle.dump(embed_dictt,f)
f.close()