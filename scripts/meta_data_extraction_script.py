import sys
import os
import logging

#logging.basicConfig(level=logging.WARNING) # this line makes you see warnings or hiher only in the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_features_extraction.meta_features_extraction import process_meta_features
from experiment_manager.models import KnowledgeBaseExperiment_V2

process_meta_features(table=KnowledgeBaseExperiment_V2)