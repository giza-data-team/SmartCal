from itertools import product

from config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from config.enums.calibration_hyperparameters import CalibrationHyperparameters


def get_calibration_hyperparameters(cal_algo: CalibrationAlgorithmTypesEnum):
    """
    Retrieves the hyperparameters configuration for a given calibration algorithm.
    
    Maps each calibration algorithm to its specific hyperparameters using the
    CalibrationHyperparameters enum values.
    
    Args:
        cal_algo (CalibrationAlgorithmTypesEnum): The calibration algorithm
        
    Returns:
        dict: Hyperparameters configuration for the algorithm
                Empty dict if algorithm has no hyperparameters
    """

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
        CalibrationAlgorithmTypesEnum.IMAX: {
            'imax_cal_mode': CalibrationHyperparameters.imax_cal_mode.value,
            'Q_binning_stage': CalibrationHyperparameters.Q_binning_stage.value,
            'bin_init_mode': CalibrationHyperparameters.bin_init_mode.value
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
        },
        CalibrationAlgorithmTypesEnum.MIXANDMATCH: {}
    }
    return hyperparameters.get(cal_algo, {})

def get_all_calibrator_combinations():
    """Generate all possible calibrator and hyperparameter combinations using function calls."""
    
    calibration_algorithms = list(CalibrationAlgorithmTypesEnum)
    calibrator_hyperparameters = []
    for cal in calibration_algorithms:
        calibrator_hyperparameters.append(get_calibration_hyperparameters(cal))
    
    # Generate all possible combinations as a list of dictionaries with specific handling for "META"
    combinations_dicts = []
    for algo, params in zip(calibration_algorithms, calibrator_hyperparameters):
        if algo.name == "META":  # Special handling for META
            if "alpha" in params and "acc" in params:
                for alpha_value in params["alpha"]:
                    combinations_dicts.append({"Calibration_Algorithm": algo.name, "alpha": alpha_value, "acc": None})
                for acc_value in params["acc"]:
                    combinations_dicts.append({"Calibration_Algorithm": algo.name, "alpha": None, "acc": acc_value})
        elif params:
            keys, values = zip(*params.items())
            for combination in product(*values):
                combinations_dicts.append({**{"Calibration_Algorithm": algo.name}, **dict(zip(keys, combination))})
        else:
            combinations_dicts.append({"Calibration_Algorithm": algo.name, "Hyperparameters": {}})

    return combinations_dicts