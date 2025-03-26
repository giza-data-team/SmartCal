from enum import Enum
from config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()

class CalibrationHyperparameters(Enum):
    lr = config_manager.cal_lr_tune
    max_itr = config_manager.cal_max_iter_tune
    n_bins = config_manager.cal_n_bins_tune
    num_bins = config_manager.cal_n_bins_tune
    model_type = config_manager.beta_params_tune # for beta will modify the name
    imax_cal_mode = config_manager.imax_cal_mode_tune
    imax_num_bins = config_manager.imax_num_bins
    Q_binning_stage = config_manager.imax_q_binning_stage_tune
    bin_init_mode = config_manager.imax_bin_init_mode_tune
    calibrator_type_meta = config_manager.meta_calibrator_calibrator_type_tune
    alpha = config_manager.meta_alpha_tune
    acc = config_manager.meta_acc_tune
    initial_T = config_manager.temperature_scaling_initial_t_tune
    lr_tempscaling = config_manager.cal_lr_tune
    max_iter_tempscaling = config_manager.cal_max_iter_tune
    max_depth = config_manager.probability_tree_max_depth_tune
    min_samples_leaf = config_manager.probability_tree_min_samples_leaf_tune
    calibrator_type_platt = config_manager.platt_calibrator_calibrator_type_tune
    calibrator_type_histogram = config_manager.histogram_calibrator_calibrator_type_tune
    adapt_temp_scaling_mode = config_manager.adaptive_temp_mode_tune
    adapt_temp_scaling_lr = config_manager.adaptive_temp_lr_tune
    adapt_temp_scaling_max_itr = config_manager.adaptive_temp_max_itr_tune
    adapt_temp_scaling_bins = config_manager.adaptive_temp_bins_tune
    mix_match_parametric = config_manager.mix_match_parametric_tune
    mix_match_nonparametric = config_manager.mix_match_nonparametric_tune


    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
