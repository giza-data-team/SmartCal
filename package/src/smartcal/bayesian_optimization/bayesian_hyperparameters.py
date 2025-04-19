from typing import Dict, Any

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()

lr = config_manager.bayes_lr_tune
max_itr = config_manager.bayes_max_iter_tune
temp =  config_manager.bayes_temp_tune
alpha = config_manager.bayes_alpha_tune
acc = config_manager.bayes_acc_tune
num_bins = config_manager.bayes_num_bins_tune

# Parameter mappings for categorical variables
param_mappings: Dict[str, Dict[str, Any]] = {
    'BETA': {
        'model_type': {'abm': 0, 'am': 1, 'ab': 2}
    },
    'META': {
        'calibrator_type': {'ALPHA': 0, 'ACC': 1}
    },
    'PLATT': {
        'calibrator_type': {'PLATT': 0, 'PLATTBINNER': 1, 'PLATTBINNERMARGINAL': 2}
    },
    'HISTOGRM': {
        'calibrator_type': {'HISTOGRAM': 0, 'HISTOGRAMMARGINAL': 1}
    },
    'AdaptiveTemperatureScaling': {
        'mode': {'linear': 0, 'entropy': 1, 'hybrid': 2}
    },
    'MIXANDMATCH': {
        'parametric_calibrator': {
            'TemperatureScalingCalibrator': 0,
            'PlattScalingCalibrator': 1,
            'VectorScalingCalibrator': 2,
            'MatrixScalingCalibrator': 3,
            'BetaCalibrator': 4,
            'MetaCalibrator': 5,
            'DirichletCalibrator': 6,
            'AdaptiveTemperatureScalingCalibrator': 7
        },
        'nonparametric_calibrator': {
            'IsotonicCalibrator': 0,
            'EmpiricalBinningCalibrator': 1,
            'HistogramCalibrator': 2
        }
    }
}

# Parameter spaces for optimization (using continuous ranges)
param_spaces: Dict[str, Dict[str, Any]] = {
    'EMPIRICALBINNING': {
        'n_bins': num_bins # Will be rounded to nearest integer
    },
    'TEMPERATURESCALING': {
        'initial_T': temp,
        'lr_tempscaling': lr,
        'max_iter_tempscaling': max_itr  # Will be rounded
    },
    'ISOTONIC': {},
    'BETA': {
        'model_type': (0, 2)  # Will be mapped to categorical
    },
    'DIRICHLET': {
        'lr': lr,
        'max_iter': max_itr
    },
    'META': {
        'calibrator_type': (0, 1),
        'alpha': alpha,
        'acc': acc
    },
    'MATRIXSCALING': {
        'lr': lr,
        'max_iter': max_itr
    },
    'VECTORSCALING': {
        'lr': lr,
        'max_iter': max_itr
    },
    'PLATT': {
        'calibrator_type': (0, 2),
        'num_bins': num_bins
    },
    'HISTOGRM': {
        'calibrator_type': (0, 1),
        'num_bins': num_bins
    },
    'MIXANDMATCH': {
        'parametric_calibrator': (0, 7),
        'nonparametric_calibrator': (0, 2)
    },
    'AdaptiveTemperatureScaling': {
        'initial_T': temp,
        'lr_tempscaling': lr,
        'max_iter_tempscaling': max_itr,
        'mode': (0, 2),
        'confidence_bins': num_bins,
        'entropy_bins': num_bins
    }
}
