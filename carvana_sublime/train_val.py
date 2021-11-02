from CustomDatasets import CustomDatasetForTrain, CustomDatasetForTest
from Net import NeuralNetwork
from Dice import DiceMetric, SoftDiceLoss
import torch
from my_functions import get_data_csv, get_train_test
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


if __name__ == "__main__":
	dataset_path = '/home/dima/carvana_dataset'
	imgs_path  = dataset_path + '/train/train'
	masks_path = dataset_path + '/train_masks/train_masks'

	nn_image_shape = (512, 512)
	learning_rate = 0.001
	num_epochs = 5
	mask_treashold = 0.5
	batch_size = 2

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	data = get_data_csv(imgs_path=imgs_path, masks_path=masks_path)
    
	# Добавляем признак, по которому будем разбивать датасет на train и test,
	# чтобы не было разных фотографий одной и той же машины в двух датасетах
	data["car"] = data["file_name"].apply(lambda x: x.split('_')[0])
	
	# Обучение с валидацией
	train_df, valid_df = get_train_test(data, separate_feature='car', test_size=0.25)
	train_df.reset_index(inplace=True, drop=True)
	valid_df.reset_index(inplace=True, drop=True)

	train_data = CustomDatasetForTrain(train_df, device, out_shape=nn_image_shape)
	valid_data = CustomDatasetForTrain(valid_df, device, out_shape=nn_image_shape, skip_mask=True)

	train_data_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
	valid_data_loader = DataLoader(valid_data,batch_size=batch_size, shuffle=False)
	
	# Создаем модель на основе предложенной архитектуры
	model = smp.Unet('mobilenet_v2', classes=1, encoder_depth=5, 
		         encoder_weights='imagenet', decoder_channels = [256, 128, 64, 32, 16]).to(device)

	#model = smp.Unet('mobilenet_v2', classes=1, encoder_depth=5, 
	#                 encoder_weights='imagenet').to(device)

	my_model = NeuralNetwork(model=model)
	
	criterion = SoftDiceLoss()
	optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
	metric = DiceMetric(treashold=mask_treashold)
	
	result = my_model.fit(criterion, metric, optimizer, train_data_loader, valid_data_loader, epochs=num_epochs)
	print(result)
	
