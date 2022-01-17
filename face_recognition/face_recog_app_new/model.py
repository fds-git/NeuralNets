import torch.nn as nn
import torch
import numpy as np
from facenet_pytorch import MTCNN
import pandas as pd
from typing import List, Tuple
import cv2
from mobile_imagenet import MobileImagenet


class MyFaceRecognizer(nn.Module):
    '''Класс, реализующий детекцию и распознавание лиц'''

    def __init__(self, device: str,  model_path: str, database_path: str = None):
        '''Конструктор класса
        Входные параметры:
        device: str - устройство, на котором будет выполняться модель
        model_path: str - путь до сохраненных весов модели
        database_path: str - путь до pandas pickle объекта, хранящего данные о пользователях'''

        super(MyFaceRecognizer, self).__init__()

        self.device = device
        self.detector = MTCNN(keep_all=True, device=device).eval()
        self.recognizer = MobileImagenet().to(device)
        self.recognizer.load_state_dict(torch.load(model_path))
        self.recognizer.eval()

        if database_path != None:
            try:
                # pickle сохраняет в том числе типы данных в колонках (в данном случае centroids - ndarray-объекты)
                self.database = pd.read_pickle(database_path)
                # Получаем из колонки 'centroid' датафрейма двумерный массив, содержащий координаты всех центроид,
                # а затем преобразуем его в двумерный тензор для более быстрой обработки
                centroids = np.vstack(self.database['centroid'].values)
                self.centroids = torch.from_numpy(centroids).to(device)
            except:
                self.database = None
                self.centroids = None


    def preprocessing(self, rgb_frame_numpy: np.ndarray, box: tuple) -> torch.Tensor:
        '''Метод подготовки изображения лица, ограниченного прямоугольником box на rgb_frame_numpy для подачи в сеть
        Входные параметры:
        rgb_frame_numpy: np.ndarray - изображение, на котором нужно распознать лицо
        box: tuple - прямоугольник, ограничивающий лицо на изображении rgb_frame_numpy
        Возвращаемые значения:
        face_tensor: torch.Tensor - подготовленное для подачи в сеть изображение лица в тензорном формате, 
        для которого нужно вычислить эмбэддинг'''

        left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
        face_numpy = rgb_frame_numpy[top:bottom, left:right, :]
        face_numpy = cv2.resize(face_numpy, (160, 160)).astype('float')
        face_numpy = face_numpy/255.0
        face_numpy = face_numpy - np.array([0.5904295, 0.44956186, 0.38371497])
        face_numpy = face_numpy / np.array([0.2431854, 0.20771153, 0.19577332])
        face_tensor = torch.from_numpy(face_numpy).to(self.device)
        face_tensor = face_tensor.permute(2,0,1)
        face_tensor = torch.unsqueeze(face_tensor, 0)
        face_tensor = face_tensor.float()

        return face_tensor


    def get_nearest_name(self, face_embedding: torch.Tensor, distance_treashold: float = 1.0) -> str:
        '''Метод определения имени человека, которое соответствует эмбэддингу face_embedding
        путем сравнения расстояний до известных центроид self.centroids
        Входные параметры:
        face_embedding: torch.Tensor - эмбэддинг лица человека
        distance_treashold: float = 1.0 - пороговое расстояние до центроид, при превышении которого 
        считаем человека незнакомым
        Возвращаемые значения:
        nearest_name: str - имя человека, которое соответствует face_embedding'''

        centroid_distances = torch.linalg.vector_norm((self.centroids - face_embedding), ord=2, dim=1)
        print(centroid_distances)
        idx_min = torch.argmin(centroid_distances).item()
        distance_min = torch.min(centroid_distances).item()
        if distance_min >= distance_treashold:
            nearest_name = 'unknown'
        else:
            nearest_name = self.database['name'][idx_min]
        return nearest_name


    @staticmethod
    def box_filter(boxes: List[List[float]], probs: List[float], prob_treashold: float = 0.9) -> List[List]:
        '''Статический метод фильтрации релевантных ограничивающих прямоугольников boxes
        по порогу вероятности prob_treashold
        Входные параметры:
        boxes: List[List[float]] - список, содержищий списки координат ограничивающих прямоугольников
        probs: List[float] - список, содержащий вероятности нахождения лиц в ограничивающих прямоугольниках 
        prob_treashold: float - порог вероятности, по которому будут фильтроваться ограничивающие прямоугольники
        Возвращаемые значения:
        boxes_filtered: List[List[float]] - отфильтрованный список координат ограничивающих прямоугольников
        probs_filtered: List[float] - отфильтрованный список вероятностей'''

        boxes_filtered = []
        probs_filtered = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob >= prob_treashold:
                    boxes_filtered.append(box)
                    probs_filtered.append(prob)
        return boxes_filtered, probs_filtered
