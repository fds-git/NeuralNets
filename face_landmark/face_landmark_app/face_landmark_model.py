import torch.nn as nn
import torch
import numpy as np
from facenet_pytorch import MTCNN
import pandas as pd
from typing import List, Tuple
import cv2
from XCeption import XceptionNet


class LandmarkDetector(nn.Module):
    '''Класс, реализующий детекцию ключевых точек лиц'''

    def __init__(self, device: str, face_size: tuple = (128, 128), model_path: str = None):
        '''Конструктор класса
        Входные параметры:
        device: str - устройство, на котором будет выполняться модель
        database_path: str - путь до pandas pickle объекта, хранящего данные о пользователях
        Возвращаемые значения:
        объект класса MyFaceRecognizer'''

        super(LandmarkDetector, self).__init__()

        self.device = device
        self.face_detector = MTCNN(keep_all=True, device=device).eval()
        self.landmark_detector = XceptionNet().to(device).eval()
        self.landmark_detector.load_state_dict(torch.load(model_path))
        self.face_size = face_size



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
        face_numpy = cv2.resize(face_numpy, self.face_size).astype('float')
        face_numpy = face_numpy/255.0
        face_tensor = torch.from_numpy(face_numpy).to(self.device)
        face_tensor = face_tensor.permute(2,0,1)
        face_tensor = torch.unsqueeze(face_tensor, 0)
        face_tensor = face_tensor.float()
        return face_tensor

    def postprocessing(self, landmarks: torch.Tensor, box: tuple) -> np.ndarray:
        left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
        landmarks = landmarks.detach().cpu().numpy()
        landmarks = landmarks.reshape(-1, 2)
        landmarks = landmarks * np.array([[(right-left)/self.face_size[0], (bottom-top)/self.face_size[1]]])
        landmarks = landmarks + np.array([[left, top]])
        landmarks = landmarks.astype(int)
        return landmarks


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