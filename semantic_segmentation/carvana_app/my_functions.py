import pandas as pd
import torch
import numpy as np
import glob
from sklearn.model_selection import train_test_split


def get_data_csv(imgs_path: str = None, masks_path: str = None) -> pd.DataFrame:
    '''Функция получает на вход пути к директориям с изображениями и масками
    и генерирует датафрейм, содержащий имя изображений, их адреса и адреса
    соответствующих им масок
    Входные параметры:
    imgs_path: str - путь к директории с изображениями,
    masks_path: str - путь к директории с масками
    Возвращаемые значения:
    pd.DataFrame: data - dataframe, содержащий адреса изображений и соответствующих им масок'''

    assert (imgs_path != None) & (masks_path != None)

    data_img = {}
    data_mask = {}
    data_img['imgs_path'] = []
    data_mask['masks_path'] = []
    data_img['imgs_path'] = list(glob.glob(imgs_path + "/*"))
    data_mask['masks_path'] = list(glob.glob(masks_path + "/*"))

    data_img = pd.DataFrame(data_img)
    data_mask = pd.DataFrame(data_mask)

    def file_name(x):
        return x.split("/")[-1].split(".")[0]

    data_img["file_name"] = data_img["imgs_path"].apply(lambda x: file_name(x))
    data_mask["file_name"] = data_mask["masks_path"].apply(lambda x: file_name(x)[:-5])

    data = pd.merge(data_img, data_mask, on = "file_name", how = "inner")

    return data
    
    
def get_train_test(source_df: pd.DataFrame, separate_feature: str = None, test_size: float = 0.25) -> pd.DataFrame:
    '''Функция разделяет source_df на две части с коэффициентом test_size
    по уникальным значениям separate_feature так, чтобы в новых датафреймах
    не было строк с одинаковыми значениями из separate_feature
    Входные параметры:
    source_df: pd.DataFrame - датафрейм для разделения на train и test
    separate_feature: str - поле, по которому датафрейм будет разделен
    test_size: float - коэффициент разделения дтафрейма
    Возвращаемые значения:
    pd.DataFrame: data_train - датафрейм для тренировки
    pd.DataFrame: data_valid - датафрейм для валидации'''
  
    if (separate_feature != None) & (separate_feature in source_df.columns):
        train_cars, valid_cars = train_test_split(source_df[separate_feature].unique(), test_size=test_size, random_state=42)
        data_valid = source_df[np.isin(source_df[separate_feature].values, valid_cars)]
        data_train = source_df[np.isin(source_df[separate_feature].values, train_cars)]
        assert source_df.shape[0] == (data_valid.shape[0] + data_train.shape[0])
        assert np.isin(data_train[separate_feature].values, data_valid[separate_feature].values).sum() == 0
    else:
        data_train, data_valid = train_test_split(source_df, test_size=test_size)

    return data_train, data_valid


def DICE(logits: torch.Tensor, targets: torch.Tensor, treashold: float) -> float:
    '''Функция для вычисления DICE коэффициента для набора изображенй в формате torch.Tensor
    Входные параметры:
    logits: torch.Tensor - тензор из предсказанных масок в logit масштабе
    targets: torch.Tensor - тензор из целевых целевых значений масок
    treashold: float - порог для определения класса точки в предсказанной точке
    Возвращаемые значения:
    score: float - значение DICE коэффициента для набора предсказанных масок'''
    
    smooth = 1
    num = targets.size(0)
    probs = torch.sigmoid(logits)
    outputs = torch.where(probs > treashold, 1, 0)
    m1 = outputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = score.sum() / num
    return score


def tensor_to_rle(tensor: torch.Tensor) -> str:
    '''Функция принимает одну маску в тензорном формате, элементы которой
    имеют значения 0. и 1. и генерирует rle представление маски в строковом формате
    Входные параметры:
    tensor: torch.Tensor - маска в тензорном формате
    Возвращаемые значения:
    rle_str: str - rle представление маски в строком виде'''
    
    # Для правильной работы алгоритма необходимо, чтобы первое и последнее значения выпрямленной маски
    # (что соответствует двум углам изображения) были равны 0. Это не должно повлиять на качество работы
    # алгоритма, так как мы не ожидаем наличие объекта в этих точках (но даже если он там будет, качество
    # не сильно упадет)
    tensor = tensor.view(1, -1)
    tensor = tensor.squeeze(0)
    tensor[0] = 0
    tensor[-1] = 0
    rle = torch.where(tensor[1:] != tensor[:-1])[0] + 2
    rle[1::2] = rle[1::2] - rle[:-1:2]
    rle = rle.cpu().detach().numpy()
    rle_str = rle_to_string(rle)
    return rle_str

    
def numpy_to_rle(mask_image: np.ndarray) -> str:
    '''Функция принимает одну маску в формате массива numpy, элементы которой
    имеют значения 0. и 1. и генерирует rle представление маски в строковом формате
    Входные параметры:
    mask_image: numpy.ndarray - маска в тензорном формате
    Возвращаемые значения:
    rle_str: str - rle представление маски в строковом виде'''
    
    # Для правильной работы алгоритма необходимо, чтобы первое и последнее значения выпрямленной маски
    # (что соответствует двум углам изображения) были равны 0. Это не должно повлиять на качество работы
    # алгоритма, так как мы не ожидаем наличие объекта в этих точках (но даже если он там будет, качество
    # не сильно упадет)
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle_str = rle_to_string(runs)
    return rle_str
 

def rle_to_string(runs: torch.Tensor) -> str:
    '''Функция преобразует последовательноть чисел в тензоре runs
    в строковое представление этой последовательности
    Входные параметры:
    runs: torch.Tensor - последовательность чисел в тензорном формате
    Возвращаемые значения:
    rle_str: str - строковое представление последовательности чисел'''
    
    rle_str = ' '.join(str(x) for x in runs)
    return rle_str


def mask_to_rle(mask_addr: str) -> str:
    '''Функция преобразует маску, имеющую адрес mask_addr и сохраненную в
    формате .gif, элементы которой имеют значения 0 и 1 в rle представление
    в строковом виде
    Входные параметры:
    mask_addr: str - адрес маски
    Возвращаемые значения:
    mask_rle: str - rle представление маски в строком виде'''
    
    mask = Image.open(mask_addr).convert('LA') # преобразование в серый
    mask = np.asarray(mask).astype('float')[:,:,0]
    mask = mask/255.0
    mask_rle = numpy_to_rle(mask)
    return mask_rle