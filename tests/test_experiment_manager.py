import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from smartcal.config.enums.dataset_types_enum import DatasetTypesEnum
from smartcal.config.enums.experiment_type_enum import ExperimentType
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from experiment_manager.experiment_manager import ExperimentManager


class TestExperimentManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dataset_type = DatasetTypesEnum.TABULAR
        self.experiment_type = ExperimentType.BENCHMARKING
        
        # Create mock DB session
        self.mock_db = MagicMock()
        
        # Create the ExperimentManager instance with mocked dependencies
        with patch('experiment_manager.experiment_manager.SessionLocal') as mock_session:
            mock_session.return_value = self.mock_db
            self.experiment_manager = ExperimentManager(
                dataset_type=self.dataset_type,
                experiment_type=self.experiment_type,
            )

    def test_initialization(self):
        """Test ExperimentManager initialization."""
        self.assertEqual(self.experiment_manager.dataset_type, DatasetTypesEnum.TABULAR)
        self.assertEqual(self.experiment_manager.experiment_type, ExperimentType.BENCHMARKING)

    @patch('pandas.read_excel')
    def test_generate_experiment(self, mock_read_excel):
        """Test _generate_experiment method."""
        # Mock the Excel data
        mock_data = {
            'Dataset': ['dataset1'],
            'Type': ['binary'],
            'no. instances': [1000],
            'no. classes': [2],
            'Experiment Type': [1]
        }
        mock_df = pd.DataFrame(mock_data)
        mock_read_excel.return_value = mock_df

        experiments = self.experiment_manager._generate_experiment()
        
        self.assertIsInstance(experiments, dict)
        self.assertIn('dataset1', experiments)
        self.assertEqual(experiments['dataset1']['no_classes'], 2)
        self.assertEqual(experiments['dataset1']['no_instances'], 1000)

    def test_fetch_next(self):
        """Test _fetch_next method."""
        # Create mock enums instead of strings
        mock_model = MagicMock()
        mock_model.name = 'model1'
        mock_cal_algo = MagicMock()
        mock_cal_algo.name = 'algo1'
        mock_metric = MagicMock()
        mock_metric.name = 'metric1'

        # Mock input data
        test_experiments = {
            'dataset1': {
                'task_type': DatasetTypesEnum.TABULAR,
                'dataset_name': 'dataset1',
                'no_classes': 2,
                'no_instances': 1000,
                'classification_type': 'binary',
                'classification_models': [mock_model],
                'calibration_algorithms': [mock_cal_algo],
                'calibration_metrics': [mock_metric],
                'dataset_path': 'path/to/dataset'
            }
        }

        # Mock database query results
        self.mock_db.query().filter().all.return_value = []

        config = self.experiment_manager._fetch_next(test_experiments)
        
        self.assertIsInstance(config, dict)
        self.assertEqual(config['dataset_name'], 'dataset1')
        self.assertEqual(config['no_classes'], 2)

    @patch('experiment_manager.experiment_manager.PipelineFactory')
    def test_run_experiment(self, mock_pipeline_factory):
        """Test _run_experiment method."""
        # Mock configuration with all required fields
        test_config = {
            'task_type': DatasetTypesEnum.TABULAR,
            'dataset_name': 'dataset1',
            'dataset_path': 'path/to/dataset',
            'combinations': {},
            'experiment_type': ExperimentType.BENCHMARKING,
            'no_classes': 2
        }

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_factory.create_pipeline.return_value = mock_pipeline

        self.experiment_manager._run_experiment(test_config)
        
        # Verify pipeline was created and run
        mock_pipeline_factory.create_pipeline.assert_called_once()
        mock_pipeline.run.assert_called_once()

    def test_save_results(self):
        """Test _save_results method."""
        # Mock results data
        test_results = {
            'experiment_type': ExperimentType.BENCHMARKING,
            'dataset_info': {
                'n_instances_cal_set': 100,
                'split_ratios(train_cal_tst)': [0.6, 0.2, 0.2]
            },
            'models_results': {
                'model1': {
                    'calibration_results': {
                        'algo1': {
                            'run_id': 1,
                            'cal_status': Experiment_Status_Enum.COMPLETED.value
                        }
                    }
                }
            }
        }

        # Mock experiment instance
        mock_experiment = MagicMock()
        self.mock_db.query().filter().first.return_value = mock_experiment

        self.experiment_manager._save_results(test_results)
        
        # Verify experiment was updated
        self.assertEqual(mock_experiment.status, Experiment_Status_Enum.COMPLETED.value)

    def test_get_experiment_data(self):
        """Test get_experiment_data method."""
        # Mock experiment data
        mock_experiment = MagicMock()
        mock_experiment.id = 1
        mock_experiment.dataset_name = 'dataset1'
        mock_experiment.status = Experiment_Status_Enum.COMPLETED.value
        
        self.mock_db.query().filter().first.return_value = mock_experiment

        result = self.experiment_manager.get_experiment_data(1)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], 1)
        self.assertEqual(result['dataset_name'], 'dataset1')
        self.assertEqual(result['status'], Experiment_Status_Enum.COMPLETED.value)

    def test_handle_pipeline_failure(self):
        """Test _handle_pipeline_failure method."""
        failure_results = {
            'run_ids': [1, 2],
            'error_message': 'Test error'
        }

        # Mock experiment instances
        mock_experiment = MagicMock()
        self.mock_db.query().filter_by().first.return_value = mock_experiment

        self.experiment_manager._handle_pipeline_failure(failure_results)
        
        # Verify experiments were updated with failure status
        self.assertEqual(mock_experiment.status, Experiment_Status_Enum.FAILED.value)
        self.assertEqual(mock_experiment.error_message, 'Test error')

    def tearDown(self):
        """Clean up after each test method."""
        self.mock_db.close()

if __name__ == '__main__':
    unittest.main()