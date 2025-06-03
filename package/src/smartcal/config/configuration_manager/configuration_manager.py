import os

from smartcal.config.configuration_manager.configuration_manager_base import ConfigurationManagerBase
from smartcal.config.configuration_manager.configuration_manager_factory import ConfigManagerFactory
from smartcal.config.enums.config_reader_enum import ConfigReaderEnum


class ConfigurationManager(ConfigurationManagerBase):
    config_reader = ConfigReaderEnum.get_value(os.getenv("CONFIG_MANAGER", "env"))
    config_manager = ConfigManagerFactory(config_reader=config_reader)
    config = config_manager.get_config()

    @property
    def n_bins(self):
        return int(self.config('N_BIN', default=10))
    
    @property
    def n_bins_calibraion_curve(self):
        return int(self.config('N_BIN_CAL_CURVE', default=5))

    @property
    def device(self):
        return str(self.config('DEVICE'))
    
    @property
    def patience(self):
        return int(self.config('PATIENCE', default=5))
    
    @property
    def min_lr_img(self):
        return float(self.config('MIN_LR_IMG'))
    
    @property
    def max_lr_img(self):
        return float(self.config('MAX_LR_IMG', default=5))
    
    @property
    def min_delta_early_stopper_img(self):
        return float(self.config('MIN_DELTA_EARLY_STOPPER_IMG', default=5))
    
    @property
    def num_itr_early_stopper_img(self):
        return int(self.config('NUM_ITR_EARLY_STOPPER_IMG', default=5))
    
    @property    
    def language_transformer_lr_min(self):
        return float(self.config('LANGUAGE_TRANSFORMER_LR_MIN'))
    
    @property    
    def language_transformer_lr_max(self):
        return float(self.config('LANGUAGE_TRANSFORMER_LR_MAX'))
    
    @property    
    def language_fasttext_lr_min(self):
        return float(self.config('LANGUAGE_FASTTEXT_LR_MIN'))
    
    @property    
    def language_fasttext_lr_max(self):
        return float(self.config('LANGUAGE_FASTTEXT_LR_MAX'))
    
    @property    
    def language_lr_finder_epochs(self):
        return int(self.config('LANGUAGE_LR_FINDER_EPOCHS'))
    
    @property    
    def language_train_epochs(self):
        return int(self.config('LANGUAGE_TRAIN_EPOCHS'))
    
    @property    
    def language_patience(self):
        return int(self.config('LANGUAGE_PATIENCE'))
    
    @property    
    def language_min_delta(self):
        return float(self.config('LANGUAGE_MIN_DELTA'))
    
    @property    
    def language_batch_size(self):
        return int(self.config('LANGUAGE_BATCH_SIZE'))
    
    @property    
    def language_monitor_metric(self):
        return str(self.config('LANGUAGE_MONITOR_METRIC'))
    
    @property
    def n_bin_cal(self):
        return int(self.config('N_BIN_CAL', default=10))
    
    @property
    def beta_model(self):
        return self.config('BETA_MODEL', 'abm')
    
    @property
    def random_seed(self):
        return int(self.config('RANDOM_SEED', default=42))

    @property
    def n_bin_metrics(self):
        return int(self.config('N_BIN_METRICS'))

    @property
    def conf_threshold(self):
        return float(self.config('CONF_THRESHOLD'))
    
    @property
    def conf_thresholds_list(self):
        thresholds = self.config('CONF_THRESHOLDS_LIST')
        threshold_strings = thresholds.strip('[]').split(',')
        return [float(threshold.strip()) for threshold in threshold_strings]

    @property
    def split_ratios(self):
        split_ratios_str = self.config('SPLIT_RATIOS', default='(0.7,0.15,0.15)')
        return tuple(map(float, split_ratios_str.strip('()').split(',')))

    @property
    def split_ratios_images(self):
        split_ratios_images_str = self.config('SPLIT_RATIOS_IMAGES', default='(0.8,0.2)')
        return tuple(map(float, split_ratios_images_str.strip('()').split(',')))

    @property
    def batch_size(self):
        return int(self.config('BATCH_SIZE', default=32))

    @property
    def img_size(self):
        return int(self.config('IMG_SIZE', default=224))

    @property
    def tokenizer_max_length(self):
        return int(self.config('TOKENIZER_MAX_LENGTH', default=512))

    @property
    def bert_tokenizer(self):
        return str(self.config('BERT_TOKENIZER', default='bert-base-uncased'))

    @property
    def config_img(self):
        return str(self.config('CONFIG_IMG'))

    @property
    def config_tabular(self):
        return str(self.config('CONFIG_TABULAR'))

    @property
    def config_language(self):
        return str(self.config('CONFIG_LANGUAGE'))
    
    @property
    def lr_tempscaling(self):
        return float(self.config('LR_TEMPSCALING'))

    @property
    def max_iter_tempscaling(self):
        return int(self.config('MAX_ITER_TEMPSCALING'))
    @property
    def lr_vectorscaling(self):
        return float(self.config('LR_VECTORSCALING'))

    @property
    def max_iter_vectorscaling(self):
        return int(self.config('MAX_ITER_VECTORSCALING'))

    @property
    def lr_matrixscaling(self):
        return float(self.config('LR_MATRIXSCALING'))

    @property
    def max_iter_matrixscaling(self):
        return int(self.config('MAX_ITER_MATRIXSCALING'))
 
    @property
    def initial_T(self):
        return float(self.config('INITIAL_T'))  
    
    @property
    def n_bin_meta_features(self):
        return int(self.config('N_BIN_META_FEATURES')) 

    @property
    def meta_data_file(self):
        return str(self.config('META_DATA_FILE'))  

    @property    
    def lr_dirichlet(self):
        return float(self.config('LR_DIRICHLET'))

    @property    
    def max_iter_dirichlet(self):
        return int(self.config('MAX_ITER_DIRICHLET'))    
    
    @property
    def epochs(self):
        return int(self.config('EPOCHS', default=5))   
    
    @property
    def learning_rate(self):
        return float(self.config('LEARNING_RATE', default=0.001))

    @property    
    def probability_tree_max_depth(self):
        return int(self.config('PROBABILITYTREE_MAX_DEPTH', default=5))

    @property    
    def probability_tree_min_samples_leaf(self):
        return int(self.config('PROBABILITYTREE_MIN_SAMPLES_LEAF', default=15))

    @property    
    def meta_alpha(self):
        return float(self.config('META_ALPHA', default=0.1))

    @property    
    def meta_acc(self):
        return float(self.config('META_ACC', default=0.85))
    
    @property    
    def meta_default_constraint(self):
        return str(self.config('META_DEFAULT_CONSTRAINT')) # can be ALPHA or ACC
    
    @property    
    def platt_default_type(self):
        return str(self.config('PLATT_DEFAULT_TYPE')) # can be PLATT, PLATTBINNER or PLATTBINNERMARGINAL
    
    @property    
    def histogram_default_type(self):
        return str(self.config('HISTOGRAM_DEFAULT_TYPE')) # can be HISTOGRAM or HISTOGRAMMARGINAL

    @property
    def n_confidence_bins(self):
        return int(self.config('N_CONFIDENCE_BINS', default=10))

    @property
    def n_entropy_bins(self):
        return int(self.config('N_ENTROPY_BINS', default=10))
    
    # FOR HYPERPARAMETER TUNING
    @property    
    def beta_params_tune(self):
        param_types = self.config('BETA_PARAMS_TUNE')
        param_types = param_types.strip("'[]'")
        return [param.strip().strip("'") for param in param_types.split(',')]

    @property    
    def cal_n_bins_tune(self):
        n_bins = self.config('CAL_NUM_BINS_TUNE')
        n_bins = n_bins.strip("'[]'")
        return [int(bin_value.strip()) for bin_value in n_bins.split(',')]
    
    @property    
    def cal_lr_tune(self):
        lr_values = self.config('CAL_LR_TUNE')
        lr_values = lr_values.strip("'[]'")
        return [float(lr.strip()) for lr in lr_values.split(',')]
    
    
    @property    
    def cal_max_iter_tune(self):
        max_iter = self.config('CAL_MAX_ITER_TUNE')
        max_iter = max_iter.strip("'[]'")
        return [int(iter_value.strip()) for iter_value in max_iter.split(',')]
    
    @property    
    def meta_calibrator_calibrator_type_tune(self):
        calibrator_types = self.config('META_CALIBRATOR_CALIBRATOR_TYPE_TUNE')
        calibrator_types = calibrator_types.strip("'[]'")
        return [type.strip().strip("'") for type in calibrator_types.split(',')]
    
    @property    
    def meta_alpha_tune(self):
        alpha_values = self.config('META_ALPHA_TUNE')
        alpha_values = alpha_values.strip("'[]'")
        return [float(alpha.strip()) for alpha in alpha_values.split(',')]
    
    @property    
    def meta_acc_tune(self):
        acc_values = self.config('META_ACC_TUNE')
        acc_values = acc_values.strip("'[]'")
        return [float(acc.strip()) for acc in acc_values.split(',')]
    
    @property    
    def probability_tree_max_depth_tune(self):
        max_depth_values = self.config('PROBABILITY_TREE_MAX_DEPTH_TUNE')
        max_depth_values = max_depth_values.strip("'[]'")
        return [None if value.strip() == 'None' else int(value.strip()) for value in max_depth_values.split(',')]
    
    @property    
    def probability_tree_min_samples_leaf_tune(self):
        min_samples_leaf = self.config('PROBABILITY_TREE_MIN_SAMPLES_LEAF_TUNE')
        min_samples_leaf = min_samples_leaf.strip("'[]'")
        return [int(value.strip()) for value in min_samples_leaf.split(',')]
    
    @property    
    def temperature_scaling_initial_t_tune(self):
        initial_t_values = self.config('TEMPERATURE_SCALING_INITIAL_T_TUNE')
        initial_t_values = initial_t_values.strip("'[]'")
        return [float(value.strip()) for value in initial_t_values.split(',')]
    
    @property    
    def histogram_calibrator_calibrator_type_tune(self):
        calibrator_types = self.config('HISTOGRAM_CALIBRATOR_CALIBRATOR_TYPE_TUNE')
        calibrator_types = calibrator_types.strip("'[]'")
        return [type.strip().strip("'") for type in calibrator_types.split(',')]
    
    @property    
    def platt_calibrator_calibrator_type_tune(self):
        calibrator_types = self.config('PLATT_CALIBRATOR_CALIBRATOR_TYPE_TUNE')
        calibrator_types = calibrator_types.strip("'[]'")
        return [type.strip().strip("'") for type in calibrator_types.split(',')]
       
    @property    
    def adaptive_temp_mode_tune(self):
        adapt_temp_modes = self.config('ADAPT_TEMP_SCALING_MODES_TUNE')
        adapt_temp_modes = adapt_temp_modes.strip("'[]'")
        return [type.strip().strip("'") for type in adapt_temp_modes.split(',')]
    
    @property    
    def adaptive_temp_bins_tune(self):
        temp_bins = self.config('ADAPT_TEMP_SCALING_BINS_TUNE')
        temp_bins = temp_bins.strip("'[]'")
        return [int(value.strip()) for value in temp_bins.split(',')]
     
    @property    
    def adaptive_temp_lr_tune(self):
        return float(self.config('ADAPT_TEMP_SCALING_lR_TUNE', default=0.1))

    @property    
    def adaptive_temp_max_itr_tune(self):
        return int(self.config('ADAPT_TEMP_SCALING_MAX_ITR_TUNE', default=100))
    
    @property    
    def mix_match_parametric_tune(self):
        parametric_cals = self.config('MIX_MATCH_PARAMETRIC')
        parametric_cals_values = parametric_cals.strip("'[]'")
        return [type.strip().strip("'") for type in parametric_cals_values.split(',')]
     
    @property    
    def mix_match_nonparametric_tune(self):
        nonparametric_cals = self.config('MIX_MATCH_NONPARAMETRIC')
        nonparametric_cals_values = nonparametric_cals.strip("'[]'")
        return [type.strip().strip("'") for type in nonparametric_cals_values.split(',')]
    
    ## For DB
    @property
    def db_host(self):
        return str(self.config('DB_HOST'))

    @property
    def db_port(self):
        return int(self.config('DB_PORT'))

    @property
    def db_name(self):
        return str(self.config('DB_NAME'))

    @property
    def db_user(self):
        return str(self.config('DB_USER'))

    @property
    def db_password(self):
        return str(self.config('DB_PASSWORD'))
    
    @property    
    def cal_tune_kfolds(self):
        return int(self.config('CAL_KFOLDS'))
    
    # Remote machine access
    @property
    def ssh_enabled(self):
        return str(self.config('SSH_ENABLED')).strip().lower() == "true"

    @property
    def ssh_host(self):
        return str(self.config('SSH_HOST'))

    @property
    def ssh_port(self):
        return int(self.config('SSH_PORT'))

    @property
    def ssh_user(self):
        return str(self.config('SSH_USER'))

    @property
    def ssh_key_path(self):
        return str(self.config('SSH_KEY_PATH'))

    @property
    def remote_bind_port(self):
        return int(self.config('REMOTE_BIND_PORT'))

    @property
    def local_bind_port(self):
        return int(self.config('LOCAL_BIND_PORT'))

    @property
    def baseline_results(self):
        return str(self.config('BASELINE_RESULTS'))

    @property
    def k_recommendations(self):
        return int(self.config('K_RECOMMENDATIONS', default=5))

    @property    
    def bayes_lr_tune(self):
        lr_values = self.config('BAYES_LR')
        lr_values = lr_values.strip("'()")  # Remove parentheses
        return tuple(float(value.strip()) for value in lr_values.split(','))  # Convert to float and return as tuple

    @property    
    def bayes_max_iter_tune(self):
        max_itr_values = self.config('BAYES_MAX_ITR')
        max_itr_values = max_itr_values.strip("'()")  # Remove parentheses
        return tuple(int(value.strip()) for value in max_itr_values.split(','))  # Convert to integer and return as tuple

    @property    
    def bayes_temp_tune(self):
        temp_values = self.config('BAYES_TEMP')
        temp_values = temp_values.strip("'()")  # Remove parentheses
        return tuple(float(value.strip()) for value in temp_values.split(','))  # Convert to float and return as tuple

    @property    
    def bayes_alpha_tune(self):
        alpha_values = self.config('BAYES_ALPHA')
        alpha_values = alpha_values.strip("'()")  # Remove parentheses
        return tuple(float(value.strip()) for value in alpha_values.split(','))  # Convert to float and return as tuple

    @property    
    def bayes_acc_tune(self):
        acc_values = self.config('BAYES_ACC')
        acc_values = acc_values.strip("'()")  # Remove parentheses
        return tuple(float(value.strip()) for value in acc_values.split(','))  # Convert to float and return as tuple
    
    @property
    def bayes_num_bins_tune(self):
        bins_values = self.config('BAYES_NUM_BINS')
        bins_values = bins_values.strip("'()")  # Remove parentheses
        return tuple(float(value.strip()) for value in bins_values.split(','))  # Convert to float and return as tuple
    
    @property
    def meta_models_path(self):
        return str(self.config('META_MODELS_PATH'))

    @property
    def meta_model_type(self):
        """Get the currently selected meta model type"""
        return str(self.config('META_MODEL_TYPE', default='xgb')).lower()

    @property
    def meta_model_version(self):
        return str(self.config('META_MODEL_VERSION'))

    @property
    def n_splits(self):
        return int(self.config('N_SPLITS', default=5))

    @property
    def results_dir(self):
        return str(self.config('RESULTS_DIR'))
    
    @property
    def meta_model_k(self):
        return int(self.config('META_MODEL_K', default=5))
    
    @property
    def num_itr_eval(self):
        num_itr_values = self.config('NUM_ITR_EVAL')
        num_itr_values = num_itr_values.strip("'[]'")
        return [int(value.strip()) for value in num_itr_values.split(',')] 
    
    @property
    def metric(self):
        return str(self.config('METRIC'))

    @property
    def supported_metrics(self):
        supported_metrics = self.config('SUPPORTED_METRICS')
        supported_metrics_values = supported_metrics.strip("'[]'")
        return [type.strip().strip("'") for type in supported_metrics_values.split(',')]
    
    @property
    def logging(self):
        return str(self.config('LOGGING')).strip().lower() == "true"

    @property
    def meta_label_encoder_path(self):
        return str(self.config('META_LABEL_ENCODER_PATH'))

    @property
    def insights_folder_path(self):
        return str(self.config('INSIGHTS_FOLDER_PATH'))

    @property
    def all_class_distributions_file(self):
        return str(self.config('ALL_CLASS_DISTRIBUTIONS_FILE'))
