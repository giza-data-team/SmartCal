import unittest
import numpy as np
import torch
from scipy.stats import entropy
from calibration_algorithms.imax import ImaxCalibrator

class TestImaxCalibrator(unittest.TestCase):
	def setUp(self):
		# Common test data
		self.num_classes = 5
		self.num_samples = 1000
		np.random.seed(42)

		self.logits = np.random.randn(self.num_samples, self.num_classes).astype(np.float32)
		self.labels = np.random.randint(0, self.num_classes, size=self.num_samples)

		# Test configurations
		self.base_config = {
			"imax_cal_mode": "Top1",
			"imax_num_bins": 15,
			"Q_binning_stage": "raw"
		}

	def test_basic_functionality(self):
		"""Test basic calibration workflow"""
		calibrator = ImaxCalibrator(**self.base_config)

		# Test fitting
		calibrator.fit(self.logits, self.labels)
		self.assertTrue(calibrator.fitted)

		# Test prediction shape and validity
		test_logits = np.random.randn(10, self.num_classes)
		calibrated = calibrator.predict(test_logits)
		self.assertEqual(calibrated.shape, test_logits.shape)
		self.assertTrue(np.all(calibrated >= 0))
		self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-3))

	def test_scw_grouping(self):
		"""Test sCW class grouping with known priors"""
		calibrator = ImaxCalibrator(imax_cal_mode="SCW")

		# Force known class distribution
		class_priors = np.array([0.05, 0.06, 0.58, 0.59, 0.9])
		calibrator.class_priors = torch.from_numpy(class_priors)

		# Test grouping
		calibrator._form_class_groups()
		expected_groups = [[0, 1], [2, 3], [4]]
		self.assertEqual(calibrator.class_groups, expected_groups)

	def test_mutual_information(self):
		"""Test mutual information increases during training"""
		calibrator = ImaxCalibrator(**self.base_config)
		calibrator.fit(self.logits, self.labels)

		# Check MI history
		mi_history = calibrator.mi_history
		self.assertGreater(len(mi_history), 0)

		# Verify MI increases (allow small fluctuations)
		for i in range(1, len(mi_history)):
			self.assertGreaterEqual(mi_history[i], mi_history[i - 1] - 1e-6)

	def test_cw_calibration(self):
		"""Test class-wise calibration"""
		calibrator = ImaxCalibrator(imax_cal_mode="CW")
		calibrator.fit(self.logits, self.labels)

		# Verify per-class bins
		self.assertEqual(calibrator.bin_boundaries.shape[0], self.num_classes)
		self.assertEqual(calibrator.bin_probs.shape[0], self.num_classes)

	def test_top1_calibration(self):
		"""Test top1 calibration strategy"""
		calibrator = ImaxCalibrator(imax_cal_mode="Top1")
		calibrator.fit(self.logits, self.labels)

		# Verify only top1 is calibrated
		test_logits = np.random.randn(10, self.num_classes)
		calibrated = calibrator.predict(test_logits)

		top1_mask = np.zeros_like(calibrated, dtype=bool)
		top1_indices = np.argmax(test_logits, axis=1)
		top1_mask[np.arange(10), top1_indices] = True

		self.assertTrue(np.all(calibrated[top1_mask] > 0))
		self.assertTrue(np.all(calibrated[~top1_mask] == 0))

	def test_temperature_scaling(self):
		"""Test temperature scaling preprocessing"""
		calibrator = ImaxCalibrator(Q_binning_stage="scaled")
		processed = calibrator._preprocess_logits(torch.tensor(self.logits))

		# Verify scaling
		expected = torch.tensor(self.logits) / 1.0  # Temperature=1.0
		self.assertTrue(torch.allclose(processed, expected))

	def test_edge_cases(self):
		"""Test edge cases and error handling"""
		# Empty input
		with self.assertRaises(ValueError):
			calibrator = ImaxCalibrator()
			calibrator.fit(np.array([]), np.array([]))

		# Single class
		calibrator = ImaxCalibrator()
		calibrator.fit(np.random.randn(10, 1), np.zeros(10, dtype=int))
		self.assertEqual(calibrator.bin_boundaries.shape[0], 1)

	def test_numerical_stability(self):
		"""Test extreme input values"""
		# Near-zero probabilities
		logits = np.array([[100, 0, 0, 0, 0]] * 1000)
		labels = np.zeros(1000, dtype=int)

		calibrator = ImaxCalibrator()
		calibrator.fit(logits, labels)

		calibrated = calibrator.predict(logits)
		self.assertFalse(np.any(np.isnan(calibrated)))

	def test_bin_initialization(self):
		"""Test bin initialization strategy"""
		calibrator = ImaxCalibrator()
		test_logits = torch.randn(1000, 1)
		calibrator._initialize_bins(test_logits)

		# Verify quantile-based initialization
		sorted_logits = torch.sort(test_logits[:, 0]).values
		expected = torch.quantile(sorted_logits,
		                          torch.linspace(0, 1, calibrator.cfg["imax_num_bins"] + 1))
		self.assertTrue(torch.allclose(calibrator.bin_boundaries[0], expected))

	def test_consistency_with_theory(self):
		"""Test key paper claims"""
		calibrator = ImaxCalibrator(imax_num_bins=15, imax_cal_mode="CW")

		# Create perfectly calibrated data
		perfect_probs = np.eye(self.num_classes)[self.labels]
		perfect_logits = np.log(perfect_probs + 1e-10)

		calibrator.fit(perfect_logits, self.labels)

		# Verify perfect calibration
		calibrated = calibrator.predict(perfect_logits)

		self.assertTrue(np.allclose(perfect_probs, calibrated, atol=0.1))

