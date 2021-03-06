### Приложение для распознавания лиц

Проект реализован на основе face_recog_app, но в качестве модели распознавания лиц использовалась не встроенная в facenet-pytorch,
а обученная в проекте face_recog_jupyter

- embeddings.py - скрипт для добавления человека в базу database.pkl с помощью изображения его лица, полученного с вебкамеры
- recognition.py - скрипт для распознавания лиц, находящихся на изображении, полученном с вебкамеры. Для распознавания используется база database.pkl
- model.py - модуль содержащий класс для распознавания  лиц (загрузку обученных моделей детекции и распознавания, предобработку кадров, фильтрацию, определение имен людей, находящихся на кадре)
- database.pkl - сохраненный датафрейм, содержащий информацию о людях, которые модель изучила с помощью embeddings.py (имена, координаты центроид)
- mobile_imagenet.py - архитектура модели
- mobile_emb128_0.26test_margin2_122000own - сохраненные веса модели, обученной на большом датасете в проекте face_recog_jupyter
- mobile_emb128_0.28test_margin2_77000mtcnn - сохраненные веса модели, обученной на малом датасете в проекте face_recog_jupyter