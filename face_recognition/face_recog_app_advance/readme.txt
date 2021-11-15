В проекте реализованы детекция и распознавание лиц с помощью библиотеки facenet-pytorch 2.5.2: https://pypi.org/project/facenet-pytorch/
Использовались наработки из соревнования по обнаружению deep fake: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
Идея проекта и структура была взята отсюда: https://projectgurukul.org/deep-learning-project-face-recognition-with-python-opencv/
Здесь разобрано, как правильно работать с батчами с MTCNN https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch

embeddings.py - скрипт для добавления человека в базу с помощью изображения его лица, полученного с вебкамеры
recognition.py - скрипт для распознавания находящихся в базе лиц на изображении, полученном с вебкамеры
sample_mtcnn.py - скрипт, в котором разорана только детекция лиц
Check_NNs.ipynb - ноутбук для проверки качества работы нейронной сети