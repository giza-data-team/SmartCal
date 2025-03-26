from abc import ABC, abstractmethod
from config.singleton import SingletonABC


class ConfigurationManagerBase(ABC, metaclass=SingletonABC):
    """
    Configurations
    """
    @property
    @abstractmethod
    def n_bins(self):
        pass
    
    @property
    @abstractmethod
    def n_bins_calibraion_curve(self):
        pass
    
    @property
    @abstractmethod
    def random_seed(self):
        pass

    @property
    @abstractmethod
    def n_bin_cal(self):
        pass
    
    @property
    @abstractmethod
    def beta_model(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass
    
    @property
    @abstractmethod
    def patience(self):
        pass
    
    @property
    @abstractmethod
    def min_lr_img(self):
        pass
    
    @property
    @abstractmethod
    def max_lr_img(self):
        pass
    
    @property
    @abstractmethod
    def min_delta_early_stopper_img(self):
        pass
    
    @property
    @abstractmethod
    def num_itr_early_stopper_img(self):
        pass
    
    @property 
    @abstractmethod   
    def language_train_epochs(self):
        pass
    
    @property 
    @abstractmethod   
    def language_patience(self):
        pass
    
    @property   
    @abstractmethod 
    def language_min_delta(self):
        pass
    
    @property  
    @abstractmethod  
    def language_batch_size(self):
        pass
    
    @property  
    @abstractmethod  
    def language_monitor_metric(self):
        pass

    @property
    @abstractmethod
    def conf_threshold(self):
        pass
    
    @property
    @abstractmethod
    def n_bin_metrics(self):
        pass

    @property
    @abstractmethod
    def conf_threshold(self):
        pass
    
    @property
    @abstractmethod
    def conf_thresholds_list(self):
        pass
    
    @property
    @abstractmethod
    def epochs(self):
        pass
    
    @property
    @abstractmethod
    def learning_rate(self):
        pass

    @property
    @abstractmethod
    def split_ratios(self):
        pass

    @property
    @abstractmethod
    def split_ratios_images(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def img_size(self):
        pass

    @property
    @abstractmethod
    def tokenizer_max_length(self):
        pass

    @property
    @abstractmethod
    def bert_tokenizer(self):
        pass

    @property
    @abstractmethod
    def config_img(self):
        pass

    @property
    @abstractmethod
    def config_tabular(self):
        pass

    @property
    @abstractmethod
    def config_language(self):
        pass
    @property
    @abstractmethod
    def lr_tempscaling(self):
        pass

    @property
    @abstractmethod
    def max_iter_tempscaling(self):
        pass

    @property
    @abstractmethod
    def lr_vectorscaling(self):
        pass

    @property
    @abstractmethod
    def max_iter_vectorscaling(self):
        pass

    @property
    @abstractmethod
    def lr_matrixscaling(self):
        pass

    @property
    @abstractmethod
    def max_iter_matrixscaling(self):
        pass

    @property
    @abstractmethod
    def initial_T(self):
        pass

    @property
    @abstractmethod
    def n_bin_meta_features(self):
        pass

    @property
    @abstractmethod
    def meta_data_file(self):
        pass

    @property    
    @abstractmethod
    def lr_dirichlet(self):
        pass

    @property    
    @abstractmethod
    def max_iter_dirichlet(self):
        pass

    @property
    @abstractmethod
    def imax_cal_mode(self):
        pass

    @property
    @abstractmethod
    def imax_num_bins(self):
        pass

    @property
    @abstractmethod
    def imax_binning_stage(self):
        pass

    @property
    @abstractmethod
    def imax_binning_repr_scheme(self):
        pass

    @property
    @abstractmethod
    def imax_bin_init_mode(self):
        pass

    @property    
    @abstractmethod
    def probability_tree_max_depth(self):
        pass

    @property    
    @abstractmethod
    def probability_tree_min_samples_leaf(self):
        pass

    @property    
    @abstractmethod
    def meta_alpha(self):
        pass

    @property    
    @abstractmethod
    def meta_acc(self):
        pass

    @property    
    @abstractmethod
    def meta_default_constraint(self):
        pass

    @property 
    @abstractmethod   
    def platt_default_type(self):
        pass
    
    @property  
    @abstractmethod  
    def histogram_default_type(self):
        pass
    
    # FOR HYPERPARAMETER TUNING
    @property    
    @abstractmethod
    def beta_params_tune(self):
        pass
    
    @property    
    @abstractmethod
    def cal_n_bins_tune(self):
        pass
    
    @property    
    @abstractmethod
    def cal_lr_tune(self):
        pass
    
    @property    
    @abstractmethod
    def imax_cal_mode_tune(self):
        pass
    
    @property    
    @abstractmethod
    def imax_q_binning_stage_tune(self):
        pass
    
    @property    
    @abstractmethod
    def imax_bin_init_mode_tune(self):
        pass
    
    @property    
    @abstractmethod
    def cal_max_iter_tune(self):
        pass
    
    @property    
    @abstractmethod
    def meta_calibrator_calibrator_type_tune(self):
        pass

    @property    
    @abstractmethod
    def meta_alpha_tune(self):
        pass
    
    @property    
    @abstractmethod
    def meta_acc_tune(self):
        pass
    
    @property    
    @abstractmethod
    def probability_tree_max_depth_tune(self):
        pass
    
    @property    
    @abstractmethod
    def probability_tree_min_samples_leaf_tune(self):
        pass
    
    @property    
    @abstractmethod
    def temperature_scaling_initial_t_tune(self):
        pass
    
    @property    
    @abstractmethod
    def histogram_calibrator_calibrator_type_tune(self):
        pass
    
    @property    
    @abstractmethod
    def platt_calibrator_calibrator_type_tune(self):
        pass

    @property    
    @abstractmethod
    def adaptive_temp_mode_tune(self):
        pass  
    
    @property    
    @abstractmethod
    def adaptive_temp_mode_tune(self):
        pass  
    
    @property    
    @abstractmethod
    def adaptive_temp_bins_tune(self):
        pass  
    
    @property    
    @abstractmethod
    def adaptive_temp_lr_tune(self):
        pass  

    @property    
    @abstractmethod
    def adaptive_temp_max_itr_tune(self):
        pass  
        
    @property    
    @abstractmethod
    def mix_match_parametric_tune(self):
        pass  
    
    @property    
    @abstractmethod
    def mix_match_nonparametric_tune(self):
        pass  
    
    @property    
    @abstractmethod
    def cal_tune_kfolds(self):
        pass

    @property
    @abstractmethod
    def n_confidence_bins(self):
        pass

    @property
    @abstractmethod
    def n_entropy_bins(self):
        pass

    # FOR BAYESIAN OPTIMIZATION
    @property
    @abstractmethod
    def bayes_lr_tune(self):
        pass
    
    @property
    @abstractmethod
    def bayes_max_iter_tune(self):
        pass
    
    @property
    @abstractmethod
    def bayes_temp_tune(self):
        pass
    
    @property
    @abstractmethod
    def bayes_alpha_tune(self):
        pass
    
    @property
    @abstractmethod
    def bayes_acc_tune(self):
        pass
    