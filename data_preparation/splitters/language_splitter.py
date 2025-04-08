from .data_splitter_base import DatasetSplitter
from Package.src.SmartCal.utils.timer import time_operation

class LanguageSplitter(DatasetSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = {} 
    @time_operation
    def split_dataset(self, data):
        target_col = self.target_cols[self.dataset_name]
        return self.split_structured_data(data, target_col)
    
        
    def get_timing(self):
        return self.timing