from CustomDatasets import CustomDatasetForTrain, CustomDatasetForTest
from Net import NeuralNetwork
from Dice import DiceMetric, SoftDiceLoss, BCESoftDiceLoss
import torch
from my_functions import get_data_csv, get_train_test
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


if __name__ == "__main__":
	dataset_path = '/home/dima/datasets/carvana_dataset'
	imgs_path  = dataset_path + '/train/train'
	masks_path = dataset_path + '/train_masks/train_masks'

	batch_size = 2
	learning_rate = 0.0005
	num_epochs = 30
	mask_treashold = 0.5

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = get_data_csv(imgs_path=imgs_path, masks_path=masks_path)

	train_transform = A.Compose([
	A.Resize(1024, 2048, cv2.INTER_AREA),
    A.HorizontalFlip(p=0.5),
    ToTensorV2(),
    ])

	valid_transform = A.Compose([
    A.Resize(1024, 2048, cv2.INTER_AREA),
    ToTensorV2(),
    ])
    
	# Добавляем признак, по которому будем разбивать датасет на train и test,
	# чтобы не было разных фотографий одной и той же машины в двух датасетах
	data["car"] = data["file_name"].apply(lambda x: x.split('_')[0])
	
	# Обучение с валидацией
	train_df, valid_df = get_train_test(data, separate_feature='car', test_size=0.25)
	train_df.reset_index(inplace=True, drop=True)
	valid_df.reset_index(inplace=True, drop=True)

	train_data = CustomDatasetForTrain(train_df, device, train_transform, skip_mask=False)
	valid_data = CustomDatasetForTrain(valid_df, device, valid_transform, skip_mask=True)

	train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
	
	# Создаем модель на основе предложенной архитектуры
	model = smp.DeepLabV3Plus(encoder_name='timm-mobilenetv3_small_100', encoder_depth=5, encoder_weights='imagenet',
		encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
		in_channels=3, classes=1, activation=None, upsampling=4, aux_params=None).to(device)


	my_model = NeuralNetwork(model=model)
	
	criterion = BCESoftDiceLoss()
	optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
	metric = DiceMetric(treashold=mask_treashold)
	
	result = my_model.fit(criterion,
		metric,
		optimizer,
		train_data_loader,
		valid_data_loader,
		epochs=num_epochs)

	my_model.trace_save(path_to_save = './model_with_val.pt')