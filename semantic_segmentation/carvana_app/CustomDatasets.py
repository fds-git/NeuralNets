from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2

class CustomDatasetForTrain(Dataset):
    '''Класс для создания тренировочных и валидационных датасетов'''
    
    def __init__(self, data_info: pd.DataFrame, device: str, transform: object, skip_mask: bool=False):
        '''Входные параметры:
        data_info: pd.DataFrame - датафрейм с адресами изображений и масок
        device: str - имя устройства, на котором будут обрабатываться данные
        transform: object - список преобразований, которым будут подвергнуты изображения и маски
        skip_mask: bool - флаг, нужно ли генерировать исходную маску (без изменения размерности)'''
        
        # Подаем подготовленный датафрейм
        self.data_info = data_info
        # Разделяем датафрейм на rgb картинки 
        self.image_arr = self.data_info.iloc[:,0]
        # и на сегментированные картинки
        self.mask_arr = self.data_info.iloc[:,2]
        # Количество пар картинка-сегментация
        self.data_len = len(self.data_info.index)
        # Устройство, на котором будут находиться выходные тензоры
        self.device = device
        # Нужно ли пробрасывать маску изображения на выход без изменений
        self.skip_mask = skip_mask
        # Сохраняем преобразования данных
        self.transform = transform

        
    def __getitem__(self, index: int):
        '''Входные параметры:
        index: int - индекс для обращения к элементам датафрейма data_info
        Возвращаемые значения:
        tr_image: torch.Tensor - тензорное представление изображения
        tr_mask: torch.Tensor - тензорное представление маски
        mask: torch.Tensor - тензорное представление маски без преобразований
        (возвращается если значение skip_mask равно True - необходимо при валидации)'''
        
        image = cv2.imread(self.image_arr[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float')/255.0
        
        # gif не открывается через open cv, поэтому используем для чтения PIL Image
        mask = Image.open(self.mask_arr[index])
        mask = np.asarray(mask)#.astype('float')
        
        transformed = self.transform(image=image, mask=mask)
        tr_image = transformed['image']
        tr_mask = transformed['mask']
        
        tr_image = tr_image.to(self.device).float()
        tr_mask = tr_mask.to(self.device).float().unsqueeze(0)
        
        # Если необходима исходная маска, то дополнительно возвращаем ее
        if self.skip_mask == True:
            mask = (torch.as_tensor(mask)).to(self.device).float().unsqueeze(0)
            return (tr_image, tr_mask, mask)
        else:
            return (tr_image, tr_mask)

        
    def __len__(self):
        return self.data_len


class CustomDatasetForTest(Dataset):
    '''Класс для создания тестовых датасетов'''
    
    def __init__(self, data_info, device: str, transform: object):
        '''Входные параметры:
        data_info: pd.DataFrame - датафрейм с адресами изображений и масок
        device: str - имя устройства, на котором будут обрабатываться данные
        transform: object - список преобразований, которым будут подвергнуты изображения и маски
        Возвращаемые значения:
        объект класса CustomDatasetForTest'''
        
        # Подаем наш подготовленный датафрейм
        self.data_info = data_info
        # Получаем адреса RGB изображений 
        self.image_addresses = self.data_info.iloc[:,0]
        # Получаем имена RGB изображений 
        self.image_names = self.data_info.iloc[:,1]
        # Количество пар картинка-сегментация
        self.data_len = len(self.data_info.index)
        # Устройство, на котором будут находиться выходные тензоры
        self.device = device
        # Сохраняем преобразования данных
        self.transform = transform

        
    def __getitem__(self, index):
        '''Входные параметры:
        index: int - индекс для обращения к элементам датафрейма data_info
        Возвращаемые значения:
        index: int - индекс для обращения к элементам датафрейма data_info
        tr_image: torch.Tensor - тензорное представление изображения
        image_name: str - имя изображения'''
        
        image = cv2.imread(self.image_addresses[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float')/255.0
        
        transformed = self.transform(image=image)
        tr_image = transformed['image']
        tr_image = tr_image.to(self.device).float()
        image_name = self.image_names[index]
    
        return (index, tr_image, image_name)

    
    def __len__(self):
        return self.data_len
        

