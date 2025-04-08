from Package.src.SmartCal.config.enums.dataset_types_enum import DatasetTypesEnum
from Package.src.SmartCal.config.enums.image_models_enum import ImageModelsEnum
from Package.src.SmartCal.config.enums.language_models_enum import LanguageModelsEnum
from Package.src.SmartCal.config.enums.tabular_models_enum import TabularModelsEnum
from Package.src.SmartCal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.config.enums.calibration_hyperparameters import CalibrationHyperparameters
from Package.src.SmartCal.config.enums.experiment_type_enum import ExperimentType
from .models import BenchmarkingExperiment, KnowledgeBaseExperiment, BenchmarkingExperiment_V2, KnowledgeBaseExperiment_V2
config_manager = ConfigurationManager()

class ExperimentConfig:
    @staticmethod
    def get_experiment_class(experiment_type: ExperimentType):
        mapping = {
            ExperimentType.BENCHMARKING: BenchmarkingExperiment,
            ExperimentType.KNOWLEDGE_BASE: KnowledgeBaseExperiment,
            ExperimentType.BENCHMARKING_V2: BenchmarkingExperiment_V2,
            ExperimentType.KNOWLEDGE_BASE_V2: KnowledgeBaseExperiment_V2
        }
        if experiment_type not in mapping:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        return mapping[experiment_type]



    @staticmethod
    def get_model_type(dataset_type: DatasetTypesEnum):
        """
        Retrieves the list of models available for the given dataset type.
        
        Args:
            dataset_type (DatasetTypesEnum): Type of dataset (Image, Language, Tabular)
        
        Returns:
            list: List of model enums for the given dataset type.
        """
        mapping = {
            DatasetTypesEnum.IMAGE: ImageModelsEnum,
            DatasetTypesEnum.LANGUAGE: LanguageModelsEnum,
            DatasetTypesEnum.TABULAR: TabularModelsEnum
        }
        return list(mapping[dataset_type])
    
    
    @staticmethod
    def get_config_path(dataset_type: DatasetTypesEnum):
        """
        Retrieves the configuration file path for a given dataset type.
        
        Args:
            dataset_type (DatasetTypesEnum): Type of dataset (Image, Language, Tabular)
        
        Returns:
            str: Path to the configuration file.
        """
        mapping = {
            DatasetTypesEnum.IMAGE: config_manager.config_img,
            DatasetTypesEnum.LANGUAGE: config_manager.config_language,
            DatasetTypesEnum.TABULAR: config_manager.config_tabular
        }
        return mapping[dataset_type]
    
    
    @staticmethod
    def get_calibration_hyperparameters(cal_algo: CalibrationAlgorithmTypesEnum, use_defaults=False):
        """
        Retrieves the hyperparameters configuration for a given calibration algorithm.

        Maps each calibration algorithm to its specific hyperparameters using the
        CalibrationHyperparameters enum values.

        Args:
            cal_algo (CalibrationAlgorithmTypesEnum): The calibration algorithm
            use_defaults (bool): If True, use default values for Beta and Temperature scaling


        Returns:
            dict: Hyperparameters configuration for the algorithm
                  Empty dict if algorithm has no hyperparameters
        """

        if use_defaults and cal_algo == CalibrationAlgorithmTypesEnum.BETA:
            return {}

        if use_defaults and cal_algo == CalibrationAlgorithmTypesEnum.TEMPERATURESCALING:
            return {}

        hyperparameters = {
            CalibrationAlgorithmTypesEnum.BETA: {
                'model_type': CalibrationHyperparameters.model_type.value
            },
            CalibrationAlgorithmTypesEnum.EMPIRICALBINNING: {
                'n_bins': CalibrationHyperparameters.n_bins.value
            },
            CalibrationAlgorithmTypesEnum.DIRICHLET: {
                'lr': CalibrationHyperparameters.lr.value,
                'max_iter': CalibrationHyperparameters.max_itr.value
            },
            CalibrationAlgorithmTypesEnum.MATRIXSCALING: {
                'lr': CalibrationHyperparameters.lr.value,
                'max_iter': CalibrationHyperparameters.max_itr.value
            },
            CalibrationAlgorithmTypesEnum.META: {
                'calibrator_type': CalibrationHyperparameters.calibrator_type_meta.value,
                'alpha': CalibrationHyperparameters.alpha.value,
                'acc': CalibrationHyperparameters.acc.value
            },
            CalibrationAlgorithmTypesEnum.TEMPERATURESCALING: {
                'initial_T': CalibrationHyperparameters.initial_T.value,
                'lr_tempscaling': CalibrationHyperparameters.lr_tempscaling.value,
                'max_iter_tempscaling': CalibrationHyperparameters.max_iter_tempscaling.value
            },
            CalibrationAlgorithmTypesEnum.VECTORSCALING: {
                'lr': CalibrationHyperparameters.lr.value,
                'max_iter': CalibrationHyperparameters.max_itr.value
            },
            CalibrationAlgorithmTypesEnum.ISOTONIC: {},
   
            CalibrationAlgorithmTypesEnum.PLATT: {
                'calibrator_type': CalibrationHyperparameters.calibrator_type_platt.value,
                'num_bins': CalibrationHyperparameters.num_bins.value
            },
            CalibrationAlgorithmTypesEnum.HISTOGRM: {
                'calibrator_type': CalibrationHyperparameters.calibrator_type_histogram.value,
                'num_bins': CalibrationHyperparameters.num_bins.value
            },
            CalibrationAlgorithmTypesEnum.AdaptiveTemperatureScaling: {
                'mode': CalibrationHyperparameters.adapt_temp_scaling_mode.value,
                'confidence_bins': CalibrationHyperparameters.adapt_temp_scaling_bins.value,
                'entropy_bins': CalibrationHyperparameters.adapt_temp_scaling_bins.value,
                'initial_T': CalibrationHyperparameters.initial_T.value,
                'lr_tempscaling': CalibrationHyperparameters.lr.value
            },
            CalibrationAlgorithmTypesEnum.MIXANDMATCH: {
                'parametric_calibrator': CalibrationHyperparameters.mix_match_parametric.value,
                'nonparametric_calibrator': CalibrationHyperparameters.mix_match_nonparametric.value
            }
        }
        return hyperparameters.get(cal_algo, {})