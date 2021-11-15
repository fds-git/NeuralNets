import face_recognition
import cv2
import numpy as np
import glob
import time
import csv
import pickle


f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)
f.close()

f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)
f.close()

known_face_encodings = []
known_face_names = []

for ref_id , embed_list in embed_dictt.items():
	for embed in embed_list:
		known_face_encodings +=[embed]
		known_face_names += [ref_id]
   												


video_capture = cv2.VideoCapture(0)
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True  :
	ret, frame = video_capture.read()
	# Уменьшаем размер изображения для ускорения
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	# Меняем порядок каналов (OpenCV используем BGR, face_recognition - RGB)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Обрабатываем каждый второй кадр
	if process_this_frame:
		# Находим все лица и их энкодинги для текущего кадра видео
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		face_names = []

		# Обрабатываем скрытое представление каждого лица
		for face_encoding in face_encodings:
			# Функция вернет True для лиц known_face_encodings, удаленных от face_encoding менее чем на 0.6 - по умолчанию
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
			name = "Unknown"

			# # Если совпадения есть, берем индекс самого первого
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Либо берем лицо из базы, имеющее минимальное расстояние до текущего лица
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			print(face_distances)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
			face_names.append(name)

	process_this_frame = not process_this_frame

	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Обратное преобразование координат для соответствия исходному (неуменьшенному изображению)
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Подписываем имя
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	font = cv2.FONT_HERSHEY_DUPLEX

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()