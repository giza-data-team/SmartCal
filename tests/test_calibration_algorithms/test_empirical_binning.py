import unittest
import numpy as np

from smartcal.calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator


class TestEmpiricalBinningCalibrator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        self.n_bins = 5
        self.calibrator = EmpiricalBinningCalibrator(n_bins=self.n_bins)
        
        # Create sample test data in 2D format
        np.random.seed(42)
        probs = np.random.random(100)
        self.sample_predictions = np.column_stack([1 - probs, probs])
        self.sample_ground_truth = np.random.randint(0, 2, 100)

    def test_initialization(self):
        """Test proper initialization of the calibrator."""
        self.assertEqual(self.calibrator.n_bins, self.n_bins)
        self.assertIsNone(self.calibrator.bin_boundaries)
        self.assertIsNone(self.calibrator.bin_probabilities)
        self.assertFalse(self.calibrator.fitted)

    def test_validate_init_params(self):
        """Test parameter validation during initialization."""
        # Test invalid n_bins type
        with self.assertRaises(TypeError):
            EmpiricalBinningCalibrator(n_bins=5.5)
        
        # Test negative n_bins
        with self.assertRaises(ValueError):
            EmpiricalBinningCalibrator(n_bins=-1)
        
        # Test zero n_bins
        with self.assertRaises(ValueError):
            EmpiricalBinningCalibrator(n_bins=0)

    def test_validate_input(self):
        """Test input validation functionality."""
        # Test valid 2D input
        pred_2d = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        gt = np.array([0, 1, 0])
        validated_pred, validated_gt = self.calibrator.validate_input(pred_2d, gt)
        self.assertEqual(validated_pred.shape, (3,2))
        self.assertTrue(np.array_equal(validated_gt, gt))

        # Test invalid shape
        with self.assertRaises(ValueError):
            self.calibrator.validate_input(np.array([0.5, 0.3, 0.2]))

        # Test invalid prediction values
        with self.assertRaises(ValueError):
            self.calibrator.validate_input(np.array([[1.5, -0.5]]))

        # Test invalid ground truth values
        with self.assertRaises(ValueError):
            self.calibrator.validate_input(
                np.array([[0.3, 0.7], [0.4, 0.6]]), 
                np.array([0, 2])
            )

        # Test length mismatch
        with self.assertRaises(ValueError):
            self.calibrator.validate_input(
                np.array([[0.3, 0.7], [0.4, 0.6]]), 
                np.array([0])
            )

    def test_compute_bin_statistics(self):
        """Test bin statistics computation."""
        predictions = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], 
                              [0.3, 0.7], [0.1, 0.9]])
        ground_truth = np.array([0, 0, 1, 1, 1])
        
        # First validate and convert to 1D
        pred_1d, gt = self.calibrator.validate_input(predictions, ground_truth)
        
        boundaries, probabilities = self.calibrator.compute_bin_statistics(
            pred_1d, gt
        )
        
        # Check shapes
        self.assertEqual(len(boundaries), self.n_bins + 1)
        self.assertEqual(len(probabilities), self.n_bins)
        
        # Check value ranges
        self.assertTrue(np.all(boundaries >= 0) and np.all(boundaries <= 1))
        self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1))

    def test_fit(self):
        """Test fitting functionality."""
        self.calibrator.fit(self.sample_predictions, self.sample_ground_truth)
        
        self.assertTrue(self.calibrator.fitted)
        self.assertIsNotNone(self.calibrator.bin_boundaries)
        self.assertIsNotNone(self.calibrator.bin_probabilities)
        
        # Check shapes
        self.assertEqual(len(self.calibrator.bin_boundaries), self.n_bins + 1)
        self.assertEqual(len(self.calibrator.bin_probabilities), self.n_bins)

    def test_predict(self):
        """Test prediction functionality."""
        # First fit the calibrator
        self.calibrator.fit(self.sample_predictions, self.sample_ground_truth)
        
        # Test predictions
        predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        calibrated = self.calibrator.predict(predictions)
        
        # Check shape
        self.assertEqual(calibrated.shape, (3, 2))
        
        # Check probability constraints
        self.assertTrue(np.all(calibrated >= 0) and np.all(calibrated <= 1))
        self.assertTrue(np.allclose(np.sum(calibrated, axis=1), 1))

    def test_predict_without_fit(self):
        """Test prediction without fitting first."""
        with self.assertRaises(ValueError):
            self.calibrator.predict(np.array([[0.5, 0.5]]))

    def test_end_to_end(self):
        """Test complete workflow from fit to predict."""
        # Fit
        self.calibrator.fit(self.sample_predictions, self.sample_ground_truth)
        
        # Create new predictions in 2D format
        probs = np.random.random(50)
        new_predictions = np.column_stack([1 - probs, probs])
        
        # Predict
        calibrated = self.calibrator.predict(new_predictions)
        
        # Verify results
        self.assertEqual(calibrated.shape, (50, 2))
        self.assertTrue(np.all(calibrated >= 0) and np.all(calibrated <= 1))
        self.assertTrue(np.allclose(np.sum(calibrated, axis=1), 1))

    def test_input_output_shapes(self):
        """Explicitly test all input and output shapes"""
        # Test 2D input
        pred_2d = np.random.random((100, 2))
        pred_2d = pred_2d / pred_2d.sum(axis=1)[:, np.newaxis]  # Normalize
        gt = np.random.randint(0, 2, 100)
        
        self.calibrator.fit(pred_2d, gt)
        output = self.calibrator.predict(pred_2d)
        self.assertEqual(output.shape, (100, 2))

    def test_value_ranges(self):
        """Explicitly test value ranges and constraints"""
        # Fit calibrator
        self.calibrator.fit(self.sample_predictions, self.sample_ground_truth)
        
        # Test edge cases in 2D format
        edge_cases = np.array([
            [1.0, 0.0],
            [0.999, 0.001],
            [0.5, 0.5],
            [0.001, 0.999],
            [0.0, 1.0]
        ])
        calibrated = self.calibrator.predict(edge_cases)
        
        # Test no negative values
        self.assertTrue(np.all(calibrated >= 0), "Found negative probabilities")
        
        # Test upper bound
        self.assertTrue(np.all(calibrated <= 1), "Found probabilities > 1")
        
        # Test probability sum
        self.assertTrue(np.allclose(np.sum(calibrated, axis=1), 1),
                       "Probabilities don't sum to 1")

    def test_calibration_effectiveness(self):
        """Test if calibration actually improves probability estimates"""
        # Create synthetic data with known bias
        np.random.seed(42)
        biased_probs = np.random.beta(2, 5, 1000)
        predictions = np.column_stack([1 - biased_probs, biased_probs])
        true_labels = np.random.binomial(1, 0.5, 1000)
        
        # Fit calibrator
        self.calibrator.fit(predictions, true_labels)
        
        # Get calibrated predictions
        calibrated = self.calibrator.predict(predictions)
        
        def compute_mean_confidence(probs):
            return np.mean(np.max(probs, axis=1))
        
        def compute_accuracy(probs, labels):
            predicted_classes = np.argmax(probs, axis=1)
            return np.mean(predicted_classes == labels)
        
        # Original predictions
        original_confidence = compute_mean_confidence(predictions)
        original_accuracy = compute_accuracy(predictions, true_labels)
        
        # Calibrated predictions
        calibrated_confidence = compute_mean_confidence(calibrated)
        calibrated_accuracy = compute_accuracy(calibrated, true_labels)
        
        # The difference between confidence and accuracy should be smaller after calibration
        original_gap = abs(original_confidence - original_accuracy)
        calibrated_gap = abs(calibrated_confidence - calibrated_accuracy)
        
        self.assertLessEqual(calibrated_gap, original_gap + 0.1, 
                            "Calibration did not improve confidence-accuracy gap")

    def test_monotonicity(self):
        """Test if calibration preserves relative ordering of predictions"""
        # Create ordered predictions in 2D format
        probs = np.linspace(0.1, 0.9, 10)
        ordered_pred = np.column_stack([1 - probs, probs])
        random_labels = np.random.randint(0, 2, 10)
        
        # Fit and predict
        self.calibrator.fit(ordered_pred, random_labels)
        calibrated = self.calibrator.predict(ordered_pred)
        
        # Check if ordering is preserved
        calibrated_probs = calibrated[:, 1]  # Positive class probabilities
        self.assertTrue(np.all(np.diff(calibrated_probs) >= -1e-10), 
                       "Calibration did not preserve monotonicity")

    def test_validate_input_multiclass(self):
        predictions = np.array([
            [0.1, 0.3, 0.6],  # Probabilities for 3 classes
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1]
        ])

        ground_truth = np.array([2, 1, 0])  # Multiclass labels {0, 1, 2}

        preds, gt = self.calibrator.validate_input(predictions, ground_truth)

        np.testing.assert_array_equal(predictions, preds)  # Ensure predictions remain unchanged
        np.testing.assert_array_equal(ground_truth, gt)  # Ensure ground truth remains unchanged

    def test_multiclass_prediction(self):
        """Test calibration on multiclass predictions."""
        # Generate 3-class softmax-like probabilities
        np.random.seed(22)
        raw_preds = np.random.random((100, 3))
        normalized_preds = raw_preds / raw_preds.sum(axis=1, keepdims=True)  # Normalize to sum 1
        ground_truth = np.random.randint(0, 3, 100)  # 3-class labels

        # Fit calibrator
        self.calibrator.fit(normalized_preds, ground_truth)

        # Predict
        calibrated_preds = self.calibrator.predict(normalized_preds)

        # Check output shape
        self.assertEqual(calibrated_preds.shape, (100, 3))

        # Ensure probability constraints
        self.assertTrue(np.all(calibrated_preds >= 0))
        self.assertTrue(np.all(calibrated_preds <= 1))
        self.assertTrue(np.allclose(np.sum(calibrated_preds, axis=1), 1), "Probabilities do not sum to 1")

    def test_compute_bin_statistics_multiclass(self):
        predictions = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        ground_truth = np.array([1, 0, 1, 0])

        bin_boundaries, bin_probabilities = self.calibrator.compute_bin_statistics(predictions, ground_truth)
        self.assertEqual(len(bin_boundaries), self.n_bins + 1)
        self.assertEqual(bin_probabilities.shape, (self.n_bins, predictions.shape[1]))
