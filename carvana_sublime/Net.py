import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    '''Класс для создания работы с нейронной сетью для семантической сегментации Carvana'''
    def __init__(self, model: object):
        '''Конструктор класса
        Входные параметры:
        model: nn.Module - последовательность слоев или модель, через которую будут проходить данные
        Возвращаемые значения: 
        объект класса NeuralNetwork'''
        super(NeuralNetwork, self).__init__()
        self.model = model

    def forward(self, input_data):
        '''Функция прямого прохода через объкт класса
        Входные параметры:
        input_data: torch.Tensor - тензорное представление изображения
        Возвращаемые значения: 
        input_data: torch.Tensor - тензорное представление маски изображения'''
        output_data = self.model(input_data)
        return output_data
    
    @staticmethod
    def tensor_to_rle(tensor: torch.Tensor) -> str:
        '''Статический метод принимает одну маску в тензорном формате, элементы которой
        имеют значения 0. и 1. и генерирует rle представление маски в строковом формате
        Входные параметры:
        tensor: torch.Tensor - маска в тензорном формате
        Возвращаемые значения:
        rle_str: str - rle представление маски в строковом виде'''
    
        # Для правильной работы алгоритма необходимо, чтобы первое и последнее значения выпрямленной маски
        # (что соответствует двум углам изображения) были равны 0. Это не должно повлиять на качество работы
        # алгоритма, так как мы не ожидаем наличие объекта в этих точках (но даже если он там будет, качество
        # не сильно упадет)
        with torch.no_grad():
            tensor = tensor.view(1, -1)
            tensor = tensor.squeeze(0)
            tensor[0] = 0
            tensor[-1] = 0
            rle = torch.where(tensor[1:] != tensor[:-1])[0] + 2
            rle[1::2] = rle[1::2] - rle[:-1:2]
            rle = rle.cpu().detach().numpy()
            rle_str = NeuralNetwork.rle_to_string(rle)
            return rle_str
    
    @staticmethod
    def rle_to_string(runs: torch.Tensor) -> str:
        '''Функция преобразует последовательноть чисел в тензоре runs
        в строковое представление этой последовательности
        Входные параметры:
        runs: torch.Tensor - последовательность чисел в тензорном формате
        Возвращаемые значения:
        rle_str: str - строковое представление последовательности чисел'''
        return ' '.join(str(x) for x in runs)
    
    
    def fit(self, criterion: object, metric: object, optimizer: object, 
                  train_data_loader: DataLoader, valid_data_loader: DataLoader=None, epochs: int=1):
        '''Метод для обучения объекта класса
        Входные параметры:
        criterion: object - объект для вычисления loss
        metric: object - объект для вычисления метрики качества
        optimizer: object - оптимизатор
        train_data_loader: DataLoader - загрузчик данных для обучения
        valid_data_loader: DataLoader - загрузчик данных для валидации
        epochs: int - количество эпох обучения
        
        Возвращаемые значения:
        result: dict - словарь со значениями loss при тренировке, валидации и метрики при валидации 
        для каждой эпохи'''
        
        self.optimizer = optimizer
        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_valid_metrics = []
        result = {}
        
        for epoch in range(epochs):
            self.model.train()
            time1 = time.time()
            running_loss =0.0
            train_losses = []
            for batch_idx, (data, labels) in enumerate(train_data_loader):
                data, labels = Variable(data), Variable(labels)        

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_losses.append(loss.item())
                if (batch_idx+1) % 300 == 0:
                    print(f'Train Epoch: {epoch+1}, Loss: {(running_loss/300):.6f}')
                    time2 = time.time()
                    print(f'Spend time for {300*data.shape[0]} images: {(time2-time1):.6f} sec')
                    time1 = time.time()
                    running_loss = 0.0

            train_loss = np.mean(train_losses)        
            
            
            if valid_data_loader != None:
                self.model.eval()
                valid_metrics = []
                valid_losses = []
                for batch_idx, (data, labels_small, labels) in enumerate(valid_data_loader):
                    data, labels, labels_small = Variable(data), Variable(labels), Variable(labels_small)
                    outputs = self.model(data)
                    # loss вычисляется для сжатых масок для правильной валидации (обучались на сжатых)
                    # чтобы вовремя определить переобучение
                    loss = criterion(outputs, labels_small)
                    valid_losses.append(loss.item())
                    #Преобразуем выход модели к размеру соответствующей маски
                    outputs = F.interpolate(input=outputs, size=(labels.shape[2], 
                                                                 labels.shape[3]), mode='bilinear', align_corners=False)

                    # метрика считается для исходных размеров потому что именно так итоговое качество
                    # определяется алгоритмом kaggle 
                    metric_value = metric(outputs, labels)
                    valid_metrics.append(metric_value.item())
                    
                valid_loss    = np.mean(valid_losses)
                valid_metric  = np.mean(valid_metrics)
                print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}, valid_loss: {(valid_loss):.6f}, valid_metric: {(valid_metric):.6f}')
            else:
                print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}')
                valid_loss = None
                valid_metric = None
            
            epoch_train_losses.append(train_loss)
            epoch_valid_losses.append(valid_loss)
            epoch_valid_metrics.append(valid_metric)
        
        result['epoch_train_losses'] = epoch_train_losses
        result['epoch_valid_losses'] = epoch_valid_losses
        result['epoch_valid_metrics'] = epoch_valid_metrics
        
        return result
    
    def valid(self, criterion: object, metric: object, valid_data_loader: DataLoader):
        '''Метод для валидации модели
        Входные параметры:
        criterion: object - объект для вычисления loss
        metric: object - объект для вычисления метрики качества
        valid_data_loader: DataLoader - загрузчик данных для валидации
        
        Возвращаемые значения:
        result: dict - словарь со значениями loss при тренировке, валидации и метрики при валидации 
        для каждой эпохи'''
        self.model.eval()
        valid_metrics = []
        valid_losses = []
        result = {}
        for batch_idx, (data, labels_small, labels) in enumerate(valid_data_loader):
            data, labels, labels_small = Variable(data), Variable(labels), Variable(labels_small)
            outputs = self.model(data)
            # loss вычисляется для сжатых масок для правильной валидации (обучались на сжатых)
            # чтобы вовремя определить переобучение
            loss = criterion(outputs, labels_small)
            valid_losses.append(loss.item())
            #Преобразуем выход модели к размеру соответствующей маски
            outputs = F.interpolate(input=outputs, size=(labels.shape[2], 
                                                         labels.shape[3]), mode='bilinear', align_corners=False)

            # метрика считается для исходных размеров потому что именно так итоговое качество
            # определяется алгоритмом kaggle 
            metric_value = metric(outputs, labels)
            valid_metrics.append(metric_value.item())
                    
        valid_loss    = np.mean(valid_losses)
        valid_metric  = np.mean(valid_metrics)
        result['valid_loss'] = valid_loss
        result['valid_metric'] = valid_metric
        return result

    
    def predict(self, test_data_loader: DataLoader, predict_directory: str=None, output_size: tuple=(1280, 1918), 
                mask_treashold: float=0.5, generate_rle_dataframe: bool=True) -> pd.DataFrame:
        '''Метод для предсказания масок для набора изображения
        Входные параметры:
        test_data_loader: DataLoader - загрузчик данных для предсказания
        predict_directory: str - директория, в которую будут сохраняться сгенерированные маски (если None,
        то маски сохраняться не будут)
        output_size: tuple - пространственная размерность выходных масок
        mask_treashold: float - порог, по которому будет определяться класс каждой точки для масок
        generate_rle_dataframe: bool - флаг, нужна ли генерация rle представлений масок
        Возвращаемые значения:
        rle_dataframe: pd.DataFrame - датафрейм с rle представлениями для масок (если 
        generate_rle_dataframe==True)
        Маски в формате .gif для изображений с соответствующими именами, находятся в директории predict_directory'''
        self.model.eval()
        img_names = []
        img_rles = []
        time1 = time.time()
        time2 = time.time()
        for batch_idx, (index, img, img_name)  in enumerate(test_data_loader):

            img = Variable(img)        
            pred_mask_logit = self.model(img)
            pred_mask_logit = F.interpolate(input=pred_mask_logit, size=output_size, mode='bilinear', align_corners=False)
            pred_mask_logit_prob = torch.sigmoid(pred_mask_logit)
            pred_mask = torch.where(pred_mask_logit_prob > mask_treashold, 1, 0)
            
            # Каждое изображение в тензоре преобразуем в картинку и сохраняем
            for i in range(pred_mask.shape[0]):
                if predict_directory != None:
                    mask = (pred_mask[i].cpu().numpy() * 255.0)[0] # [0] - избавляемся от батч размерности
                    PIL_image = Image.fromarray(mask.astype('uint8'), 'L')
                    PIL_image.save((predict_directory+img_name[i]).split('.')[0]+'.gif')
                
                # Если требуется, получаем значения rle для каждой картинки
                if generate_rle_dataframe == True:
                    img_names.append(img_name[i])
                    img_rles.append(NeuralNetwork.tensor_to_rle(pred_mask[i]))
            
            if (batch_idx+1) % 300 == 0:
                    print('-'*50)
                    print(f'Processed images: {(batch_idx+1)*img.shape[0]}')
                    time3 = time.time()
                    print(f'Total time: {(time3-time1):.2f} sec')
                    print(f'Time to process {300*img.shape[0]} images: {(time3-time2):.2f} sec')
                    time2 = time.time()
                
        if generate_rle_dataframe == True:
            rle_dataframe = pd.DataFrame(list(zip(img_names, img_rles)), columns =['img_name', 'img_rle'])
            return rle_dataframe
    
    def save(self, path_to_save: str='./model.pth'):
        '''Метод сохранения весов модели
        Входные параметры:
        path_to_save: str - директория для сохранения состояния модели'''
        torch.save(self.model.state_dict(), path_to_save)
    
    def trace_save(self, path_to_save: str='./model.pth'):
        '''Метод сохранения модели через torchscript
        Входные параметры:
        path_to_save: str - директория для сохранения модели'''
        example_forward_input = torch.rand(1, 3, 512, 512).to('cpu')
        if next(self.model.parameters()).is_cuda:
            example_forward_input= example_forward_input.to('cuda:0')
            
        traced_model = torch.jit.trace((self.model).eval(), example_forward_input)
        torch.jit.save(traced_model, path_to_save)
    
    def onnx_save(self, path_to_save: str='./carvana_model.onnx'):
        '''Метод сохранения модели в формате ONNX
        Входные параметры:
        path_to_save: str - директория для сохранения модели'''
        example_forward_input = torch.randn(1, 3, 1024, 1024, requires_grad=True).to('cpu')
        if next(self.model.parameters()).is_cuda:
            example_forward_input= example_forward_input.to('cuda:0')

        torch.onnx.export(self.model,
                          example_forward_input,
                          path_to_save,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names = ['input'],
                          output_names = ['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},    # Модель будет работать с произвольным
                                        'output' : {0 : 'batch_size'}})  # размером батча
    
    def load(self, path_to_model: str='./model.pth'):
        '''Метод загрузки весов модели
        Входные параметры:
        path_to_model: str - директория с сохраненными весами модели'''
        self.model.load_state_dict(torch.load(path_to_model))
