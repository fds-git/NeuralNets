from CustomDatasets import CustomDatasetForTrain, CustomDatasetForTest
from Net import NeuralNetwork
from Dice import DiceMetric, SoftDiceLoss, BCESoftDiceLoss
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
	num_epochs = 30
	mask_treashold = 0.5

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = get_data_csv(imgs_path=imgs_path, masks_path=masks_path)

	# Обучение без валидации
	train_data = CustomDatasetForTrain(data, device, out_shape=nn_image_shape)
	train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True)

	
	# Создаем модель на основе предложенной архитектуры
	model = smp.Unet('efficientnet-b5', classes=1, encoder_depth=5, 
		         encoder_weights='imagenet', decoder_channels = [256, 128, 64, 32, 16]).to(device)


	my_model = NeuralNetwork(model=model)
	
	criterion = SoftDiceLoss()
	optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
	metric = DiceMetric(treashold=mask_treashold)
	
	my_model.fit(criterion,
             metric,
             optimizer,
             train_data_loader,
             epochs=num_epochs)
       
	# Сохраняем оттрассированную модель
	my_model.trace_save(path_to_save = './model_05_10.pt')
