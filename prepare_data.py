import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

def prepare_data(config):
	print("preparing data")
	df = pd.read_csv(config.data.csv_file_path)
	# print(df.head())
	df['label'] = pd.factorize(df['sentiment'])[0]
	print(df.head())

	test_size = config.data.test_set_ratio
	train_df, test_df = train_test_split(df, test_size = test_size, stratify=df["sentiment"], random_state=42)

	train_df.to_csv(config.data.train_csv_save_path, index=False)
	test_df.to_csv(config.data.test_csv_save_path, index=False)
	
if __name__ == "__main__":
	config = OmegaConf.load(r"./params.yaml")
	prepare_data(config)
