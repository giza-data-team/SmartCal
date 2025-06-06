import logging
import json
from itertools import product
import pandas as pd
from datetime import datetime
import os
import random

from sqlalchemy import or_
from experiment_manager.db_connection import SessionLocal
from smartcal.config.enums.experiment_type_enum import ExperimentType
from smartcal.config.enums.dataset_types_enum import DatasetTypesEnum
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.config.enums.language_models_enum import ModelType
from smartcal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.calibration_metrics_enum  import CalibrationMetricsEnum
from pipeline.pipeline_factory import PipelineFactory 
from experiment_manager.experiment_config import ExperimentConfig


class ExperimentManager:
    
    def __init__(self,dataset_type: DatasetTypesEnum,experiment_type: ExperimentType):
        """
        Initilization for all needed resources by the Experiment Manager

        Args
        Dataset type : ex. Image , Tabular , Language

        Experiment_type : ex. Benchmarking , KnowledgeBase 
        """
        self.dataset_type = dataset_type
        self.experiment_type = experiment_type
        self.config_manager = ConfigurationManager()
        self.exp_config = ExperimentConfig()
        self.db = SessionLocal()

    def close(self):
        """
        Call this method when you're finished using the ExperimentManager 
        to close the session properly.
        """
        self.db.close()
        logging.info(f"db session closed successfully")

    
    def _generate_experiment(self):
        """
        Generates initial experiment configurations for all datasets of a given type.
        
        This method:
        1. Loads dataset information from configuration files
        2. Filters datasets based on experiment type
        3. Creates base configurations for each dataset
        4. Adds model and calibration algorithm combinations
                    
        Returns:
            dict: Mapping of dataset names to their experiment configurations        
        """
        try:
            file_path = self.exp_config.get_config_path(self.dataset_type)

            problem_sheet = pd.read_excel(file_path)

            if problem_sheet is None:
                raise ValueError(f"No sheet found for problem type: {self.dataset_type}")
            
            # Filter datasets based on ExperimentType
            if self.experiment_type == ExperimentType.BENCHMARKING:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 1]
            elif self.experiment_type == ExperimentType.KNOWLEDGE_BASE:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 0]
            elif self.experiment_type == ExperimentType.BENCHMARKING_V2:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 3]
            elif self.experiment_type == ExperimentType.KNOWLEDGE_BASE_V2:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 2]
            
            # Convert filtered data to records - use the entire filtered sheet
            datasets = filtered_sheet[['Dataset', 'Type', 'no. instances', 'no. classes', 'Experiment Type']].to_dict('records')
            
            models = self.exp_config.get_model_type(self.dataset_type)
            calibration_algorithms = list(CalibrationAlgorithmTypesEnum)
            calibration_metrics = list(CalibrationMetricsEnum)

            experiments_by_dataset = {}
            for dataset in datasets:
                # Create base configuration for each dataset
                base_config = {
                    "task_type": self.dataset_type,
                    "dataset_name": str(dataset['Dataset']),
                    "no_classes": int(dataset['no. classes']),
                    "no_instances": int(dataset['no. instances']),
                    "classification_type": dataset['Type'],
                    "classification_models": models,
                    "calibration_algorithms": calibration_algorithms,
                    "calibration_metrics": calibration_metrics
                }
                
                base_config['dataset_path'] = f"datasets/datasets/{self.dataset_type.name}/{dataset['Dataset']}.csv"

                experiments_by_dataset[dataset['Dataset']] = base_config
                
        except Exception as e:
            logging.info(f"Error in generate_experiment: {str(e)}")
            raise
        return experiments_by_dataset
        

    
    def _fetch_next(self, experiments_by_dataset: dict, add_failed: bool = False, skip_pending: bool = False):
        ExperimentClass = self.exp_config.get_experiment_class(self.experiment_type)
        
        experiment_by_dataset = experiments_by_dataset.copy()

        # Here i randomize dataset order (to reduce chance of collision)
        dataset_names = list(experiment_by_dataset.keys())
        random.shuffle(dataset_names)
        
        for dataset_name in dataset_names:
            dataset_config = experiment_by_dataset[dataset_name]
            with self.db.begin():
                # If skip_pending is True, check if there are any pending experiments for this dataset
                if skip_pending:
                    pending_count = self.db.query(ExperimentClass).filter(
                        ExperimentClass.dataset_name == dataset_name,
                        ExperimentClass.status == Experiment_Status_Enum.PENDING.value
                    ).count()
                    
                    if pending_count > 0:
                        logging.info(f"Dataset {dataset_name} has pending experiments. Skipping.")
                        continue
                
                # Use subquery to filter existing experiments
                existing_experiments = self.db.query(ExperimentClass).filter(
                    ExperimentClass.dataset_name == dataset_name
                ).all()

                completed_combinations = set()
                failed_combinations = set()

                for exp in existing_experiments:
                    combo = (exp.classification_model, exp.calibration_algorithm, exp.calibration_metric)
                    if exp.status == Experiment_Status_Enum.COMPLETED.value:
                        completed_combinations.add(combo)
                    elif exp.status == Experiment_Status_Enum.FAILED.value:
                        failed_combinations.add(combo)

                all_combinations = list(product(
                    dataset_config["classification_models"],
                    dataset_config["calibration_algorithms"],
                    dataset_config["calibration_metrics"]
                ))

                remaining_combinations = [
                    (model, cal_algo, metric) for model, cal_algo, metric in all_combinations
                    if (model.name, cal_algo.name, metric.name) not in completed_combinations and
                    (add_failed or (model.name, cal_algo.name, metric.name) not in failed_combinations)
                ]

                if not remaining_combinations:
                    del experiments_by_dataset[dataset_name]
                    continue
                
                if remaining_combinations:
                    # For language tasks, group by model type
                    if dataset_config["task_type"] == DatasetTypesEnum.LANGUAGE:
                        bert_models = {}
                        word_embedding_models = {}
                        
                        for model, cal_algo, metric in remaining_combinations:
                            if model.model_type == ModelType.BERT:
                                if model not in bert_models:
                                    bert_models[model] = []
                                bert_models[model].append((cal_algo, metric))
                            else:  # WordEmbeddingModel
                                if model not in word_embedding_models:
                                    word_embedding_models[model] = []
                                word_embedding_models[model].append((cal_algo, metric))
                        
                        # Create separate configs for each model type
                        if bert_models:
                            experiments_by_dataset[dataset_name]["classification_models"] = [
                                m for m in dataset_config["classification_models"] if m not in bert_models
                            ]

                            if not experiments_by_dataset[dataset_name]["classification_models"]:
                                del experiments_by_dataset[dataset_name]
                            pipeline_config = dataset_config.copy()
                            pipeline_config.pop("classification_models")
                            pipeline_config.pop("calibration_algorithms")
                            pipeline_config.pop("calibration_metrics")
                            pipeline_config["combinations"] = bert_models
                            pipeline_config["experiment_type"] = self.experiment_type
                            pipeline_config["model_type"] = ModelType.BERT
                            return pipeline_config
                        
                        if word_embedding_models:
                            experiments_by_dataset[dataset_name]["classification_models"] = [
                                m for m in dataset_config["classification_models"] if m not in word_embedding_models
                            ]

                            if not experiments_by_dataset[dataset_name]["classification_models"]:
                                del experiments_by_dataset[dataset_name]

                            pipeline_config = dataset_config.copy()
                            pipeline_config.pop("classification_models")
                            pipeline_config.pop("calibration_algorithms")
                            pipeline_config.pop("calibration_metrics")
                            pipeline_config["combinations"] = word_embedding_models
                            pipeline_config["experiment_type"] = self.experiment_type
                            pipeline_config["model_type"] = ModelType.WordEmbeddingModel
                            
                            return pipeline_config
                    
                    else:
                        # For non-language tasks, proceed as normal
                        model_calibrations = {}
                        for model, cal_algo, metric in remaining_combinations:
                            if model not in model_calibrations:
                                model_calibrations[model] = []
                            model_calibrations[model].append((cal_algo, metric))

                        pipeline_config = dataset_config.copy()
                        pipeline_config.pop("classification_models")
                        pipeline_config.pop("calibration_algorithms")
                        pipeline_config.pop("calibration_metrics")
                        pipeline_config["combinations"] = model_calibrations
                        pipeline_config["experiment_type"] = self.experiment_type
                        del experiments_by_dataset[dataset_name]
                        return pipeline_config             
                
    
    def _run_experiment(self, config: dict):
        """Executes calibration experiments with metrics included."""
        experiment_ids = {}
        ExperimentClass = self.exp_config.get_experiment_class(config["experiment_type"])
        
        try:
            with self.db.begin():
                # Prepare a list of all combinations to check at once
                all_combinations = []
                for model, cal_algo_metrics in config["combinations"].items():
                    for cal_algo, metric in cal_algo_metrics:
                        all_combinations.append((
                            config["dataset_name"],
                            model.name,
                            cal_algo.name,
                            metric.name,
                            model,
                            cal_algo,
                            metric
                        ))
                
                # Batch query to find all existing experiments at once
                if all_combinations:
                    # Extract the search criteria for the query
                    search_criteria = [(c[0], c[1], c[2], c[3]) for c in all_combinations]
                    
                    # Build a query with OR conditions for each combination
                    query_conditions = []
                    for dataset_name, model_name, cal_algo_name, metric_name in search_criteria:
                        query_conditions.append(
                            (ExperimentClass.dataset_name == dataset_name) &
                            (ExperimentClass.classification_model == model_name) &
                            (ExperimentClass.calibration_algorithm == cal_algo_name) &
                            (ExperimentClass.calibration_metric == metric_name)
                        )
                    
                    existing_experiments = self.db.query(ExperimentClass).filter(
                        or_(*query_conditions)
                    ).all()
                    
                    # Create a lookup dictionary for quick access
                    existing_exp_lookup = {
                        (exp.dataset_name, exp.classification_model, exp.calibration_algorithm, exp.calibration_metric): exp
                        for exp in existing_experiments
                    }
                    
                    # Process each combination
                    for dataset_name, model_name, cal_algo_name, metric_name, model, cal_algo, metric in all_combinations:
                        # Get hyperparameters for this calibration algorithm
                        cal_hyperparams = self.exp_config.get_calibration_hyperparameters(cal_algo)
                        
                        # Check if this experiment already exists using our lookup
                        lookup_key = (dataset_name, model_name, cal_algo_name, metric_name)
                        existing_experiment = existing_exp_lookup.get(lookup_key)
                        
                        if model not in experiment_ids:
                            experiment_ids[model] = []
                        
                        if existing_experiment:
                            if existing_experiment.status == Experiment_Status_Enum.FAILED.value:
                                # Reset failed experiment to try again
                                existing_experiment.status = Experiment_Status_Enum.PENDING.value
                                existing_experiment.error_message = None
                                self.db.flush()
                            
                            # Create combination in desired format
                            combo = [
                                existing_experiment.id,
                                metric,
                                {cal_algo.value: cal_hyperparams}
                            ]
                            experiment_ids[model].append(combo)
                        else:
                            # Create new experiment record
                            experiment_config = {
                                "dataset_name": config["dataset_name"],
                                "no_classes": config["no_classes"],
                                "no_instances": config["no_instances"],
                                "classification_model": model_name,
                                "problem_type": str(config["task_type"]),
                                "classification_type": config["classification_type"],
                                "calibration_algorithm": cal_algo_name,
                                "calibration_metric": metric_name,
                                "status": Experiment_Status_Enum.PENDING.value
                            }
                            new_experiment = ExperimentClass(**experiment_config)
                            self.db.add(new_experiment)
                            self.db.flush()  # Flush to get the ID
                            
                            # Create combination in desired format
                            combo = [
                                new_experiment.id,
                                metric,
                                {cal_algo.value: cal_hyperparams}
                            ]
                            experiment_ids[model].append(combo)

            # Create final pipeline configuration
            pipeline_config = {
                "task_type": config["task_type"],
                "dataset_name": config["dataset_name"],
                "dataset_path": config["dataset_path"],
                "experiment_type": config["experiment_type"],
                "combinations": experiment_ids,
                "device": self.config_manager.device,
                "logs": True,
                "no_classes": config["no_classes"]
            }

            if config["task_type"] == DatasetTypesEnum.LANGUAGE:
                pipeline_config["model_type"] = config["model_type"]

            logging.info("\n Combinations are sent to the pipeline")
            
            # Run pipeline
            pipeline = PipelineFactory.create_pipeline(pipeline_config)
            
            # Pass the experiment manager to the pipeline for saving results
            pipeline.experiment_manager = self
            
            # Run the pipeline - results will be saved per model now
            result = pipeline.run()
            if isinstance(result, dict) and result.get('status') == Experiment_Status_Enum.FAILED.value:
                self._save_results(result)  

        except Exception as e:
            logging.error(f"Error running experiments: {str(e)}")
            raise

    def _handle_pipeline_failure(self, failure_results):
        """Handle pipeline failure by updating database records"""
        ExperimentClass = self.exp_config.get_experiment_class(self.experiment_type)
        
        try:
            with self.db.begin():
                # Update all relevant experiment records
                for run_id in failure_results.get('run_ids', []):
                    experiment = self.db.query(ExperimentClass).filter_by(id=run_id).first()
                    if experiment:
                        experiment.status = Experiment_Status_Enum.FAILED.value
                        experiment.error_message = failure_results.get('error_message')
                self.db.commit()
        except Exception as e:
            logging.error(f"Error handling pipeline failure: {str(e)}")
            raise
            
    def get_next_experiment(self, add_failed: bool = False, skip_pending: bool = False):
        """
        Get and run next available experiments from any dataset.
        
        Args:
            add_failed: Flag to include failed experiments
            skip_pending: Flag to skip datasets with pending experiments
        """
        # Generate all possible combinations for all datasets
        all_experiments = self._generate_experiment()
        while True:
            # Get next available dataset and its configurations
            next_config = self._fetch_next(all_experiments, add_failed, skip_pending)
            
            if next_config:
                logging.info(f"\n\nNext experiment for dataset: {next_config['dataset_name']}\n\n")
                self._run_experiment(next_config)
            else:
                logging.info("No new experiments available to run for any dataset")
                return None

    
    def _save_results(self,results_file: dict):
        """
        Save results from results dict to database.
        
        Args:
            results_file: Dictionary containing all results from JSON file

        
        """
        experiment_type = results_file.get('experiment_type')
        ExperimentClass = self.exp_config.get_experiment_class(experiment_type)
        
        try:
            # Check if this is a failed experiment at root level
            if results_file.get('status') == Experiment_Status_Enum.FAILED.value:
                
                experiment_ids = results_file.get('run_ids', [])
                error_message = results_file.get('error_message')
                failed_stage = results_file.get('failed_stage')
                
                with self.db.begin():
                    for exp_id in experiment_ids:
                        experiment = self.db.query(ExperimentClass).filter(
                            ExperimentClass.id == exp_id
                        ).first()
                        
                        if experiment:
                            experiment.status = Experiment_Status_Enum.FAILED.value
                            experiment.error_message = f"Failed at stage: {failed_stage}. Error: {error_message}"
                            experiment.updated_at = datetime.utcnow()
                            logging.info(f"Updated failed status for experiment {exp_id}")
                        else:
                            logging.warning(f"No experiment found with ID {exp_id}")
                return

            # Extract dataset-level information
            dataset_info = results_file.get('dataset_info', {})
            
            # Process each model's results
            model_info = results_file.get('models_results', {})
            for model_name, model_results in model_info.items():

                   
                if 'calibration_results' in model_results:
                    # Process each calibration algorithm's results
                    for cal_algo, cal_results in model_results['calibration_results'].items():
                        run_id = cal_results.get('run_id')
                        if not run_id:
                            logging.warning(f"No run_id found for {model_name} - {cal_algo}")
                            continue

                        with self.db.begin():
                            experiment = self.db.query(ExperimentClass).filter(
                                ExperimentClass.id == run_id
                            ).first()
                            
                            if not experiment:
                                logging.warning(f"No experiment found with ID {run_id}")
                                continue

                            # Update timestamp
                            experiment.updated_at = datetime.utcnow()

                            # Check calibration status
                            cal_status = cal_results.get('cal_status')
                            if cal_status == Experiment_Status_Enum.FAILED.value:
                                experiment.status = Experiment_Status_Enum.FAILED.value
                                experiment.error_message = cal_results.get('error')
                                logging.info(f"Updated results for cal_failed experiment {run_id}")
                                continue

                            # Dataset Info
                            experiment.n_instances_cal_set = dataset_info.get('n_instances_cal_set')
                            split_ratios = dataset_info.get('split_ratios(train_cal_tst)', [])
                            if len(split_ratios) == 3:
                                if isinstance(split_ratios, dict):
                                    experiment.split_ratios_train = split_ratios['train_ratio']
                                    experiment.split_ratios_cal = split_ratios['val_ratio']
                                    experiment.split_ratios_test = split_ratios['test_ratio']
                                else:
                                    experiment.split_ratios_train = split_ratios[0]
                                    experiment.split_ratios_cal = split_ratios[1]
                                    experiment.split_ratios_test = split_ratios[2]


                            
                            experiment.conf_ece_thresholds = dataset_info.get('conf_ece_thresholds')
                            
                            # Ground Truth
                            experiment.ground_truth_cal_set = dataset_info.get('ground_truth_val_set')
                            experiment.ground_truth_test_set = dataset_info.get('ground_truth_test_set')
                            
                            # Timing Information
                            preprocessing_time = dataset_info.get('preprocessing_time', {})
                            experiment.preprocessing_fit_time = preprocessing_time.get('fit_transform')
                            experiment.preprocessing_transform_time = preprocessing_time.get('transform')
                            
                            train_time = model_results.get('train_time', {})
                            experiment.train_time = train_time.get('Training_time')
                            experiment.test_time = train_time.get('Testing_time')
                            
                            cal_timing = cal_results.get('cal_timing', {})
                            experiment.calibration_fit_time = cal_timing.get('fit')
                            experiment.calibration_predict_time = cal_timing.get('predict')
                            
                            # Classification Model and Calibration Algorithm
                            experiment.classification_model = model_name.split('.')[-1]
                            experiment.calibration_algorithm = cal_algo.split('_')[0]
                            
                            # Calibration Hyperparameters
                            experiment.cal_hyperparameters = cal_results.get('best_hyperparams')
                            
                            # Uncalibrated Metrics (train set)
                            uncal_train = model_results.get('train_metrics', {})
                            experiment.uncalibrated_train_loss  = uncal_train.get('loss')
                            experiment.uncalibrated_train_accuracy = uncal_train.get('accuracy')
                            experiment.uncalibrated_train_recall_macro = uncal_train.get('recall_macro')
                            experiment.uncalibrated_train_recall_micro = uncal_train.get('recall_micro')
                            experiment.uncalibrated_train_recall_weighted = uncal_train.get('recall_weighted')
                            experiment.uncalibrated_train_precision_macro = uncal_train.get('precision_macro')
                            experiment.uncalibrated_train_precision_micro = uncal_train.get('precision_micro')
                            experiment.uncalibrated_train_precision_weighted = uncal_train.get('precision_weighted')
                            experiment.uncalibrated_train_f1_score_micro = uncal_train.get('f1_micro')
                            experiment.uncalibrated_train_f1_score_macro = uncal_train.get('f1_macro')
                            experiment.uncalibrated_train_f1_score_weighted = uncal_train.get('f1_weighted')
                            experiment.uncalibrated_train_ece = uncal_train.get('ece')
                            experiment.uncalibrated_train_mce = uncal_train.get('mce')
                            experiment.uncalibrated_train_conf_ece = uncal_train.get('conf_ece')
                            experiment.uncalibrated_train_brier_score = uncal_train.get('brier_score')
                            experiment.uncalibrated_train_calibration_curve_mean_predicted_probs = uncal_train.get('calibration_curve_mean_predicted_probs')
                            experiment.uncalibrated_train_calibration_curve_true_probs = uncal_train.get('calibration_curve_true_probs')
                            experiment.uncalibrated_train_calibration_num_bins = uncal_train.get('calibration_curve_bin_counts')
                            
                            # Uncalibrated Metrics (cal set)
                            uncal_val = model_results.get('uncalibrated_metrics_val_set', {})
                            experiment.uncalibrated_cal_loss = uncal_val.get('loss')
                            experiment.uncalibrated_cal_accuracy = uncal_val.get('accuracy')
                            experiment.uncalibrated_cal_recall_macro = uncal_val.get('recall_macro')
                            experiment.uncalibrated_cal_recall_micro = uncal_val.get('recall_micro')
                            experiment.uncalibrated_cal_precision_macro = uncal_val.get('precision_macro')
                            experiment.uncalibrated_cal_precision_micro = uncal_val.get('precision_micro')
                            experiment.uncalibrated_cal_f1_score_micro = uncal_val.get('f1_micro')
                            experiment.uncalibrated_cal_f1_score_macro = uncal_val.get('f1_macro')
                            experiment.uncalibrated_cal_ece = uncal_val.get('ece')
                            experiment.uncalibrated_cal_mce = uncal_val.get('mce')
                            experiment.uncalibrated_cal_conf_ece = uncal_val.get('conf_ece')
                            experiment.uncalibrated_cal_brier_score = uncal_val.get('brier_score')
                            experiment.uncalibrated_cal_calibration_curve_mean_predicted_probs = uncal_val.get('calibration_curve_mean_predicted_probs')
                            experiment.uncalibrated_cal_calibration_curve_true_probs = uncal_val.get('calibration_curve_true_probs')
                            experiment.uncalibrated_cal_calibration_num_bins = uncal_val.get('calibration_curve_bin_counts')
                            experiment.uncalibrated_probs_cal_set = model_results.get('uncalibrated_probs_val_set')

                            # Uncalibrated test Metrics
                            uncal_test = model_results.get('uncalibrated_metrics_test_set', {})
                            experiment.uncalibrated_test_loss = uncal_test.get('loss')
                            experiment.uncalibrated_test_accuracy = uncal_test.get('accuracy')
                            experiment.uncalibrated_test_ece = uncal_test.get('ece')
                            experiment.uncalibrated_test_mce = uncal_test.get('mce')
                            experiment.uncalibrated_test_conf_ece = uncal_test.get('conf_ece')
                            experiment.uncalibrated_test_brier_score = uncal_test.get('brier_score')
                            experiment.uncalibrated_test_calibration_curve_mean_predicted_probs = uncal_test.get('calibration_curve_mean_predicted_probs')
                            experiment.uncalibrated_test_calibration_curve_true_probs = uncal_test.get('calibration_curve_true_probs')
                            experiment.uncalibrated_test_calibration_num_bins = uncal_test.get('calibration_curve_bin_counts')
                            experiment.uncalibrated_probs_test_set = model_results.get('uncalibrated_probs_test_set')
                           
                            # Calibrated Metrics
                            if cal_status == Experiment_Status_Enum.COMPLETED.value:
                                experiment.calibration_metric = cal_results.get('calibration_metric')
                                # Validation set
                                cal_val = cal_results.get('calibrated_metrics_val_set', {})
                                experiment.calibrated_cal_loss = cal_val.get('loss')
                                experiment.calibrated_cal_accuracy = cal_val.get('accuracy')
                                experiment.calibrated_cal_ece = cal_val.get('ece')
                                experiment.calibrated_cal_mce = cal_val.get('mce')
                                experiment.calibrated_cal_conf_ece = cal_val.get('conf_ece')
                                experiment.calibrated_cal_brier_score = cal_val.get('brier_score')
                                experiment.calibrated_cal_calibration_curve_mean_predicted_probs = cal_val.get('calibration_curve_mean_predicted_probs')
                                experiment.calibrated_cal_calibration_curve_true_probs = cal_val.get('calibration_curve_true_probs')
                                experiment.calibrated_cal_calibration_num_bins = cal_val.get('calibration_curve_bin_counts')
                                experiment.calibrated_probs_cal_set = cal_results.get('calibrated_probs_val_set')
                                
                                # Test set
                                cal_test = cal_results.get('calibrated_metrics_tst_set', {})
                                experiment.calibrated_test_loss = cal_test.get('loss')
                                experiment.calibrated_test_accuracy = cal_test.get('accuracy')
                                experiment.calibrated_test_ece = cal_test.get('ece')
                                experiment.calibrated_test_mce = cal_test.get('mce')
                                experiment.calibrated_test_conf_ece = cal_test.get('conf_ece')
                                experiment.calibrated_test_brier_score = cal_test.get('brier_score')
                                experiment.calibrated_test_calibration_curve_mean_predicted_probs = cal_test.get('calibration_curve_mean_predicted_probs')
                                experiment.calibrated_test_calibration_curve_true_probs = cal_test.get('calibration_curve_true_probs')
                                experiment.calibrated_test_calibration_num_bins = cal_test.get('calibration_curve_bin_counts')
                                experiment.calibrated_probs_test_set = cal_results.get('calibrated_probs_test_set') 
                                experiment.status = Experiment_Status_Enum.COMPLETED.value
                                experiment.error_message = None
                            logging.info(f"Updated results for Completed experiment {run_id}")
                            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise
    
    def get_experiment_data(self,experiment_id: int):
        """
        Retrieve all data for a specific experiment by its ID in the same order as model definition.
        
        Args:
            experiment_id (int): ID of the experiment to retrieve
          
        
        Returns:
            dict: Flat dictionary of experiment data in model field order or None if not found
        """
        ExperimentClass = ExperimentConfig.get_experiment_class(self.experiment_type)
        
        try:
            with self.db.begin():
                experiment = self.db.query(ExperimentClass).filter(
                    ExperimentClass.id == experiment_id
                ).first()
                
                if experiment:
                    
                    # Organize data in same order as model fields
                    organized_data = {
                        # Basic Information
                        "id": experiment.id,
                        "dataset_name": experiment.dataset_name,
                        "no_classes": experiment.no_classes,
                        "no_instances": experiment.no_instances,
                        "problem_type": experiment.problem_type,
                        "classification_type": experiment.classification_type,
                        
                        # Training Configuration
                        "classification_model": experiment.classification_model,
                        "calibration_algorithm": experiment.calibration_algorithm,
                        "cal_hyperparameters": experiment.cal_hyperparameters,
                        
                        # Status and Timestamps
                        "status": experiment.status,
                        "created_at": experiment.created_at,
                        "updated_at": experiment.updated_at,
                        
                        # Dataset Split Info
                        "n_instances_cal_set": experiment.n_instances_cal_set,
                        "split_ratios_train": experiment.split_ratios_train,
                        "split_ratios_cal": experiment.split_ratios_cal,
                        "split_ratios_test": experiment.split_ratios_test,
                        
                        # Timing Information
                        "preprocessing_fit_time": experiment.preprocessing_fit_time,
                        "preprocessing_transform_time": experiment.preprocessing_transform_time,
                        "train_time": experiment.train_time,
                        "test_time": experiment.test_time,
                        "calibration_fit_time": experiment.calibration_fit_time,
                        "calibration_predict_time": experiment.calibration_predict_time,
                        
                        # Thresholds and Metrics
                        "conf_ece_thresholds": experiment.conf_ece_thresholds,
                        
                        # Uncalibrated Metrics (train set)
                        "uncalibrated_train_loss": experiment.uncalibrated_train_loss,
                        "uncalibrated_train_accuracy": experiment.uncalibrated_train_accuracy,
                        "uncalibrated_train_ece": experiment.uncalibrated_train_ece,
                        "uncalibrated_train_mce": experiment.uncalibrated_train_mce,
                        "uncalibrated_train_conf_ece": experiment.uncalibrated_train_conf_ece,
                        "uncalibrated_train_f1_score_micro": experiment.uncalibrated_train_f1_score_micro,
                        "uncalibrated_train_f1_score_macro": experiment.uncalibrated_train_f1_score_macro,
                        "uncalibrated_train_f1_score_weighted": experiment.uncalibrated_train_f1_score_weighted,
                        "uncalibrated_train_recall_micro": experiment.uncalibrated_train_recall_micro,
                        "uncalibrated_train_recall_macro": experiment.uncalibrated_train_recall_macro,
                        "uncalibrated_train_recall_weighted": experiment.uncalibrated_train_recall_weighted,
                        "uncalibrated_train_precision_micro": experiment.uncalibrated_train_precision_micro,
                        "uncalibrated_train_precision_macro": experiment.uncalibrated_train_precision_macro,
                        "uncalibrated_train_precision_weighted": experiment.uncalibrated_train_precision_weighted,
                        "uncalibrated_train_brier_score": experiment.uncalibrated_train_brier_score,
                        "uncalibrated_train_calibration_curve_mean_predicted_probs": experiment.uncalibrated_train_calibration_curve_mean_predicted_probs,
                        "uncalibrated_train_calibration_curve_true_probs": experiment.uncalibrated_train_calibration_curve_true_probs,
                        "uncalibrated_train_calibration_num_bins": experiment.uncalibrated_train_calibration_num_bins,
                        
                        # Uncalibrated Metrics (validation)
                        "uncalibrated_cal_loss": experiment.uncalibrated_cal_loss,
                        "uncalibrated_cal_recall_micro": experiment.uncalibrated_cal_recall_micro,
                        "uncalibrated_cal_recall_macro": experiment.uncalibrated_cal_recall_macro,
                        "uncalibrated_cal_precision_micro": experiment.uncalibrated_cal_precision_micro,
                        "uncalibrated_cal_precision_macro": experiment.uncalibrated_cal_precision_macro,
                        "uncalibrated_cal_f1_score_micro": experiment.uncalibrated_cal_f1_score_micro,
                        "uncalibrated_cal_f1_score_macro": experiment.uncalibrated_cal_f1_score_macro,
                        "uncalibrated_cal_accuracy": experiment.uncalibrated_cal_accuracy,
                        "uncalibrated_cal_ece": experiment.uncalibrated_cal_ece,
                        "uncalibrated_cal_mce": experiment.uncalibrated_cal_mce,
                        "uncalibrated_cal_conf_ece": experiment.uncalibrated_cal_conf_ece,
                        "uncalibrated_cal_brier_score": experiment.uncalibrated_cal_brier_score,
                        "uncalibrated_cal_calibration_curve_mean_predicted_probs": experiment.uncalibrated_cal_calibration_curve_mean_predicted_probs,
                        "uncalibrated_cal_calibration_curve_true_probs": experiment.uncalibrated_cal_calibration_curve_true_probs,
                        "uncalibrated_cal_calibration_num_bins": experiment.uncalibrated_cal_calibration_num_bins,
                        "uncalibrated_probs_cal_set": experiment.uncalibrated_probs_cal_set,
                        
                        # Uncalibrated Metrics (test)
                        "uncalibrated_test_loss": experiment.uncalibrated_test_loss,
                        "uncalibrated_test_accuracy": experiment.uncalibrated_test_accuracy,
                        "uncalibrated_test_ece": experiment.uncalibrated_test_ece,
                        "uncalibrated_test_mce": experiment.uncalibrated_test_mce,
                        "uncalibrated_test_conf_ece": experiment.uncalibrated_test_conf_ece,
                        "uncalibrated_test_brier_score": experiment.uncalibrated_test_brier_score,
                        "uncalibrated_test_calibration_curve_mean_predicted_probs": experiment.uncalibrated_test_calibration_curve_mean_predicted_probs,
                        "uncalibrated_test_calibration_curve_true_probs": experiment.uncalibrated_test_calibration_curve_true_probs,
                        "uncalibrated_test_calibration_num_bins": experiment.uncalibrated_test_calibration_num_bins,
                        "uncalibrated_probs_test_set": experiment.uncalibrated_probs_test_set,
                        
                        # Metric used for tuning
                        "calibration_metric": experiment.calibration_metric,
                        
                        # Calibrated Metrics (validation)
                        "calibrated_cal_loss": experiment.calibrated_cal_loss,
                        "calibrated_cal_accuracy": experiment.calibrated_cal_accuracy,
                        "calibrated_cal_ece": experiment.calibrated_cal_ece,
                        "calibrated_cal_mce": experiment.calibrated_cal_mce,
                        "calibrated_cal_conf_ece": experiment.calibrated_cal_conf_ece,
                        "calibrated_cal_brier_score": experiment.calibrated_cal_brier_score,
                        "calibrated_cal_calibration_curve_mean_predicted_probs": experiment.calibrated_cal_calibration_curve_mean_predicted_probs,
                        "calibrated_cal_calibration_curve_true_probs": experiment.calibrated_cal_calibration_curve_true_probs,
                        "calibrated_cal_calibration_num_bins": experiment.calibrated_cal_calibration_num_bins,
                        "calibrated_probs_cal_set": experiment.calibrated_probs_cal_set,
                        
                        # Calibrated Metrics (test)
                        "calibrated_test_loss": experiment.calibrated_test_loss,
                        "calibrated_test_accuracy": experiment.calibrated_test_accuracy,
                        "calibrated_test_ece": experiment.calibrated_test_ece,
                        "calibrated_test_mce": experiment.calibrated_test_mce,
                        "calibrated_test_conf_ece": experiment.calibrated_test_conf_ece,
                        "calibrated_test_brier_score": experiment.calibrated_test_brier_score,
                        "calibrated_test_calibration_curve_mean_predicted_probs": experiment.calibrated_test_calibration_curve_mean_predicted_probs,
                        "calibrated_test_calibration_curve_true_probs": experiment.calibrated_test_calibration_curve_true_probs,
                        "calibrated_test_calibration_num_bins": experiment.calibrated_test_calibration_num_bins,
                        "calibrated_probs_test_set": experiment.calibrated_probs_test_set,
                        
                        # Ground Truth
                        "ground_truth_test_set": experiment.ground_truth_test_set,
                        "ground_truth_cal_set": experiment.ground_truth_cal_set,
                        
                        # Error Information
                        "error_message": experiment.error_message
                    }
                    
                    # # Save to JSON file
                    # os.makedirs("db_data", exist_ok=True)
                    # json_file_path = f"db_data/experiment_{experiment_id}_{self.experiment_type.name}.json"
                    # with open(json_file_path, 'w') as json_file:
                    #     json.dump(organized_data, json_file, default=str, indent=4)
                    # logging.info(f"Experiment data saved to {json_file_path}")
                    
                    return organized_data
                else:
                    logging.info(f"No experiment found with ID {experiment_id}")
                    return None
                
        except Exception as e:
            logging.error(f"Error retrieving experiment data: {str(e)}")
            raise

