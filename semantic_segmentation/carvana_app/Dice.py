import torch.nn as nn
import torch

class DiceMetric(nn.Module):
    '''Класс для вычисления DICE коэффициента для набора изображенй в формате torch.Tensor
    с заданным порогом для определния класса каждой точки изображения'''
    
    def __init__(self, treashold: float=0.5):
        '''Входные параметры:
        treashold: float - порог для определения класса точки в предсказанной точке'''
        
        super(DiceMetric, self).__init__()
        self.treashold = treashold

        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        '''Входные параметры:
        logits: torch.Tensor - тензор из предсказанных масок в logit масштабе
        targets: torch.Tensor - тензор из целевых целевых значений масок
        Возвращаемые значения:
        score: float - значение DICE коэффициента для набора предсказанных масок'''
        
        with torch.no_grad():
            smooth = 1
            num = targets.size(0)
            probs = torch.sigmoid(logits)
            outputs = torch.where(probs > self.treashold, 1., 0.)
            m1 = outputs.view(num, -1)
            m2 = targets.view(num, -1)
            intersection = (m1 * m2)

            score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score = score.sum() / num
            return score


class SoftDiceLoss(nn.Module):
    '''Класс для вычисления DICE loss для набора изображенй в формате torch.Tensor'''
    
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        '''Входные параметры:
        logits: torch.Tensor - тензор из предсказанных масок в logit масштабе
        targets: torch.Tensor - тензор из целевых целевых значений масок
        Возвращаемые значения:
        score: float - значение DICE loss для набора предсказанных масок'''
        
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BCESoftDiceLoss(nn.Module):
    '''Класс для вычисления BCESoftDice Loss для набора изображенй в формате torch.Tensor'''
    
    def __init__(self):
        super(BCESoftDiceLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.soft_dice = SoftDiceLoss()

        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        '''Входные параметры:
        logits: torch.Tensor - тензор из предсказанных масок в logit масштабе
        targets: torch.Tensor - тензор из целевых целевых значений масок
        Возвращаемые значения:
        bce_dice: float - значение BCESoftDice loss для набора предсказанных масок'''
        
        bce_dice = self.bce(logits, targets) + self.soft_dice(logits, targets)
        return bce_dice