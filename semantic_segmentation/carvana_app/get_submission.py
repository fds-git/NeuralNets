from CustomDatasets import CustomDatasetForTest
from Net import NeuralNetwork
from torch.utils.data import DataLoader
import torch
import glob
import pandas as pd


if __name__ == "__main__":
	predict_directory = '/home/dima/carvana_dataset/test/predict_small/'
	test_dataset = '/home/dima/carvana_dataset/test/test/'
	mask_treashold = 0.5
	out_shape = (512, 512)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	# Загружаем оттрассированную модель
	my_model = torch.jit.load('./model.pt')
	my_model = NeuralNetwork(model=my_model)
	my_model = my_model.to(device)
	
	test_dataframe = {}
	test_dataframe['img_addr'] = list(glob.glob(test_dataset + "/*"))
	test_dataframe = pd.DataFrame(test_dataframe)

	test_data = CustomDatasetForTest(test_dataframe, device, out_shape)
	test_data_loader = DataLoader(test_data, batch_size=2, shuffle=False)


	rle_dataframe = my_model.predict(test_data_loader, predict_directory, 
                                 mask_treashold=mask_treashold, generate_rle_dataframe=True)
                              
	# Получаем датафрейм с результатом для заливки на kaggle
	rle_dataframe.to_csv('rle_dataframe.csv', index=True)
	sample_submission = pd.read_csv('/home/dima/carvana_dataset/sample_submission.csv')
	sample_submission = sample_submission.merge(rle_dataframe, how='left', left_on='img', right_on='img_name')
	sample_submission.drop(columns=['rle_mask', 'img_name'], inplace=True)
	sample_submission.rename(columns={'img_rle': 'rle_mask'}, inplace=True)
	sample_submission.to_csv('submission_05_10.csv', index=False)                              
