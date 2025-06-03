from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, Float, ARRAY,UniqueConstraint, Index
from datetime import datetime

from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from experiment_manager.db_connection import Base

class Experiment:
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String, nullable=False)
    no_classes = Column(Integer, nullable=False)  
    no_instances = Column(Integer, nullable=False)
    problem_type = Column(String, nullable=False)  # Image, Language, Tabular
    classification_type = Column(String, nullable=False)  # Binary, Multiclass
    
    # Training Configuration
    classification_model = Column(String, nullable=False)
    calibration_algorithm = Column(String, nullable=False)
    cal_hyperparameters = Column(JSON, nullable=True) 
    
    # Status and Timestamps
    status = Column(String, default=Experiment_Status_Enum.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dataset Split Info
    n_instances_cal_set = Column(Integer, nullable=True)
    split_ratios_train = Column(Float, nullable=True)
    split_ratios_cal = Column(Float, nullable=True)  
    split_ratios_test = Column(Float, nullable=True) 

    # Timing Information
    preprocessing_fit_time = Column(Float, nullable=True)
    preprocessing_transform_time = Column(Float, nullable=True)
    train_time = Column(Float, nullable=True)
    test_time = Column(Float, nullable=True)
    calibration_fit_time = Column(Float, nullable=True)
    calibration_predict_time = Column(Float, nullable=True)
    
    # Thresholds and Metrics
    conf_ece_thresholds = Column(ARRAY(Float), nullable=True)
    
    # Uncalibrated Metrics (train set)
    uncalibrated_train_loss = Column(Float, nullable=True)
    uncalibrated_train_accuracy = Column(Float, nullable=True)
    uncalibrated_train_ece = Column(Float, nullable=True)
    uncalibrated_train_mce = Column(Float, nullable=True)
    uncalibrated_train_conf_ece = Column(ARRAY(Float), nullable=True)
    uncalibrated_train_f1_score_macro = Column(Float, nullable=True)
    uncalibrated_train_f1_score_micro = Column(Float, nullable=True)
    uncalibrated_train_f1_score_weighted = Column(Float, nullable=True)
    uncalibrated_train_recall_macro = Column(Float, nullable=True)
    uncalibrated_train_recall_micro = Column(Float, nullable=True)
    uncalibrated_train_recall_weighted = Column(Float, nullable=True)
    uncalibrated_train_precision_macro = Column(Float, nullable=True)
    uncalibrated_train_precision_micro = Column(Float, nullable=True)
    uncalibrated_train_precision_weighted = Column(Float, nullable=True)
    uncalibrated_train_brier_score = Column(Float, nullable=True)
    uncalibrated_train_calibration_curve_mean_predicted_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_train_calibration_curve_true_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_train_calibration_num_bins = Column(ARRAY(Integer), nullable=True)
  
    # Uncalibrated Metrics (validation set)
    uncalibrated_cal_loss = Column(Float, nullable=True)
    uncalibrated_cal_recall_micro = Column(Float, nullable=True)
    uncalibrated_cal_recall_macro = Column(Float, nullable=True)
    uncalibrated_cal_precision_micro = Column(Float, nullable=True)
    uncalibrated_cal_precision_macro = Column(Float, nullable=True)
    uncalibrated_cal_f1_score_micro = Column(Float, nullable=True)
    uncalibrated_cal_f1_score_macro = Column(Float, nullable=True)
    uncalibrated_cal_accuracy = Column(Float, nullable=True)
    uncalibrated_cal_ece = Column(Float, nullable=True)
    uncalibrated_cal_mce = Column(Float, nullable=True)
    uncalibrated_cal_conf_ece = Column(ARRAY(Float), nullable=True)
    uncalibrated_cal_brier_score = Column(Float, nullable=True)
    uncalibrated_cal_calibration_curve_mean_predicted_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_cal_calibration_curve_true_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_cal_calibration_num_bins = Column(ARRAY(Integer), nullable=True)
    uncalibrated_probs_cal_set = Column(ARRAY(Float), nullable=True)  


    # Uncalibrated Metrics (test set)
    uncalibrated_test_loss = Column(Float, nullable=True)
    uncalibrated_test_accuracy = Column(Float, nullable=True)
    uncalibrated_test_ece = Column(Float, nullable=True)
    uncalibrated_test_mce = Column(Float, nullable=True)
    uncalibrated_test_conf_ece = Column(ARRAY(Float), nullable=True)
    uncalibrated_test_brier_score = Column(Float, nullable=True)
    uncalibrated_test_calibration_curve_mean_predicted_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_test_calibration_curve_true_probs = Column(ARRAY(Float), nullable=True)
    uncalibrated_test_calibration_num_bins = Column(ARRAY(Integer), nullable=True)
    uncalibrated_probs_test_set = Column(ARRAY(Float), nullable=True) 

    # Metrics used for tuning
    calibration_metric =  Column(String, nullable=True)  # ECE, MCE, ConfECE, BrierScore, etc.
    
    # Calibrated Metrics (validation Set)
    calibrated_cal_loss = Column(Float, nullable=True)
    calibrated_cal_accuracy = Column(Float, nullable=True)
    calibrated_cal_ece = Column(Float, nullable=True)
    calibrated_cal_mce = Column(Float, nullable=True)
    calibrated_cal_conf_ece = Column(ARRAY(Float), nullable=True)
    calibrated_cal_brier_score = Column(Float, nullable=True)
    calibrated_cal_calibration_curve_mean_predicted_probs = Column(ARRAY(Float), nullable=True)
    calibrated_cal_calibration_curve_true_probs = Column(ARRAY(Float), nullable=True)
    calibrated_cal_calibration_num_bins = Column(ARRAY(Integer), nullable=True)
    calibrated_probs_cal_set = Column(ARRAY(Float), nullable=True) 

    # Calibrated Metrics (Test Set)
    calibrated_test_loss = Column(Float, nullable=True)
    calibrated_test_accuracy = Column(Float, nullable=True)
    calibrated_test_ece = Column(Float, nullable=True)
    calibrated_test_mce = Column(Float, nullable=True)
    calibrated_test_conf_ece = Column(ARRAY(Float), nullable=True)
    calibrated_test_brier_score = Column(Float, nullable=True)
    calibrated_test_calibration_curve_mean_predicted_probs = Column(ARRAY(Float), nullable=True)
    calibrated_test_calibration_curve_true_probs = Column(ARRAY(Float), nullable=True)
    calibrated_test_calibration_num_bins = Column(ARRAY(Integer), nullable=True)
    calibrated_probs_test_set = Column(ARRAY(Float), nullable=True) 
    
    
    # Ground Truth
    ground_truth_test_set = Column(ARRAY(Integer), nullable=True)
    ground_truth_cal_set = Column(ARRAY(Integer), nullable=True)

    error_message = Column(Text, nullable=True)


class BenchmarkingExperiment(Base, Experiment):
    __tablename__ = "benchmarking_experiments"

    __table_args__ = (
        UniqueConstraint('dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric',
                        name='unique_experiment_combo'),
        Index('idx_dataset_name', 'dataset_name'),
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_combo', 'dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric'),
    )

class KnowledgeBaseExperiment(Base, Experiment):
    __tablename__ = 'knowledge_base_experiments'
    __table_args__ = (
        UniqueConstraint('dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric',
                        name='unique_knowledgebase_experiment'),
        Index('kb_idx_dataset_name', 'dataset_name'),
        Index('kb_idx_experiment_status', 'status'),
        Index('kb_idx_experiment_combo', 'dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric'),
    )

class BenchmarkingExperiment_V2(Base, Experiment):
    __tablename__ = 'benchmarking_experiments_V2'

    __table_args__ = (
        UniqueConstraint('dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric',
                        name='unique_benchmark_experiment_V2'),
        Index('bm_v2_idx_dataset_name', 'dataset_name'),
        Index('bm_v2_idx_experiment_status', 'status'),
        Index('bm_v2_idx_experiment_combo', 'dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric'),
    )

class KnowledgeBaseExperiment_V2(Base, Experiment):
    __tablename__ = 'knowledge_base_experiments_V2'
    __table_args__ = (
        UniqueConstraint('dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric',
                        name='unique_knowledgebase_experiment_V2'),
        Index('kb_v2_idx_dataset_name', 'dataset_name'),
        Index('kb_v2_idx_experiment_status', 'status'),
        Index('kb_v2_idx_experiment_combo', 'dataset_name', 'classification_model', 'calibration_algorithm', 'calibration_metric'),
    )