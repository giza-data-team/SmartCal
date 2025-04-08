from .preprocessor_base import Preprocessor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import ast
import numpy as np
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.utils.timer import time_operation
config_manager = ConfigurationManager()

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # convert to PIL if it's a tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        # If it's already a PIL Image, use it directly
        transformed_image = self.transform(image)
        return transformed_image, label

    def __len__(self):
        return len(self.dataset)

class ImagePreprocessor(Preprocessor):
    def __init__(self, dataset_name, metadata_path=config_manager.config_img, random_seed=config_manager.random_seed,
                 img_size=config_manager.img_size, batch_size=config_manager.batch_size, logs=False):
        super().__init__(dataset_name=dataset_name, metadata_path=metadata_path, logs=logs)
        
        self.random_seed = random_seed
        self.img_size = img_size
        self.batch_size = batch_size
        self.timing = {} 

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Load configuration using base class method
        self.config = self.load_dataset_config()
        
        # Extract stats from config
        self.stats = {
            'mean': ast.literal_eval(str(self.config['Mean'])) if isinstance(self.config['Mean'], str) else self.config['Mean'],
            'std': ast.literal_eval(str(self.config['STD'])) if isinstance(self.config['STD'], str) else self.config['STD']
        }

    def validate_config_columns(self, config):
        """Validate required columns for image preprocessing"""
        required_columns = {'Dataset', 'Torchvision_Name', 'Mean', 'STD'}
        missing_columns = required_columns - set(config.keys())
        if missing_columns:
            raise ValueError(f"Missing required columns in config: {missing_columns}")
        
        # Additional validation for the current dataset
        if self.dataset_name != config['Torchvision_Name']:
            raise ValueError(f"Dataset mismatch: {self.dataset_name} != {config['Torchvision_Name']}")

    def apply_transforms(self, dataset, transform):
        return TransformedDataset(dataset, transform)
    
    @time_operation
    def fit_transform(self, data):
        train_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.stats['mean'], std=self.stats['std'])
        ])
        transformed_dataset = self.apply_transforms(data, train_transforms)
        
        if self.logs:
            self.log_preprocessing_info(
                "Training Transforms",
                image_size=self.img_size,
                batch_size=self.batch_size,
                normalization_stats=self.stats,
                dataset_size=len(data)
            )
        
        return DataLoader(transformed_dataset, self.batch_size, shuffle=True,num_workers=4)
    
    @time_operation
    def transform(self, data):
        eval_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.stats['mean'], std=self.stats['std'])
        ])

        transformed_dataset = self.apply_transforms(data, eval_transforms)

        if self.logs:
            self.log_preprocessing_info(
                "Evaluation Transforms (Val/Testing)",
                image_size=self.img_size,
                batch_size=self.batch_size,
                normalization_stats=self.stats,
                dataset_size=len(data)
            )

        return DataLoader(transformed_dataset, self.batch_size, shuffle=False, num_workers=4)
 
    def get_timing(self):
        return self.timing