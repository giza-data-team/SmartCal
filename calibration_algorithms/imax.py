import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation
from .calibration_base import CalibratorBase
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

config_manager = ConfigurationManager()


class ImaxCalibrator(CalibratorBase):
	"""
	Complete I-MAX Calibrator implementation with:
	- JSD-based K-means initialization
	- Proper boundary management
	- Enhanced numerical stability
	- Correct mutual information calculation
	"""

	def __init__(self,
	             imax_cal_mode=config_manager.imax_cal_mode,
	             imax_num_bins=config_manager.imax_num_bins,
	             Q_binning_stage=config_manager.imax_binning_stage,
	             bin_init_mode=config_manager.imax_bin_init_mode,
	             seed=CalibratorBase.DEFAULT_SEED):
		super().__init__(seed)
		self.timing = {}
		self.cfg = {
			"imax_cal_mode": imax_cal_mode,
			"imax_num_bins": imax_num_bins,
			"Q_binning_stage": Q_binning_stage,
			"temperature": 1.0,
			"merge_threshold": 0.1,
			"em_iterations": 100,
			"bin_init_mode": bin_init_mode,
			"epsilon": 1e-10,
			"min_bin_samples": 10
		}
		self.metadata["params"].update(self.cfg)
		self.bin_boundaries = None
		self.bin_probs = None
		self.class_groups = None
		self.class_priors = None
		self.mi_history = []
		self.group_boundaries = {}  # For SCW mode
		self.group_probs = {}  # For SCW mode

	@time_operation
	def fit(self, logits: np.ndarray, labels: np.ndarray):
		logits, labels = self.validate_inputs(logits, labels)
		logits = torch.from_numpy(logits).float()
		labels = torch.from_numpy(labels).long()
		num_classes = logits.shape[1]

		# Initialize state variables
		self.bin_probs = torch.full((num_classes, self.cfg["imax_num_bins"]), 0.5)
		self.bin_boundaries = torch.zeros((num_classes, self.cfg["imax_num_bins"] + 1))
		self.class_priors = torch.zeros(num_classes)

		processed_logits = self._preprocess_logits(logits)
		self._compute_class_priors(labels)
		self._form_class_groups()
		self._initialize_bins(processed_logits)
		self._run_em_optimization(processed_logits, labels)

		self.fitted = True
		self.metadata["dataset_info"].update({
			"n_samples": logits.shape[0],
			"n_classes": num_classes,
		})

	@time_operation
	def predict(self, logits: np.ndarray) -> np.ndarray:
		if not self.fitted:
			raise RuntimeError("Calibrator not fitted. Call fit() first.")

		logits = torch.from_numpy(logits).float()
		processed_logits = self._preprocess_logits(logits)
		calibrated_probs = torch.zeros_like(processed_logits)

		if self.cfg["imax_cal_mode"] == "SCW":
			for group in self.class_groups:
				group_logits = processed_logits[:, group]
				calibrated_probs[:, group] = self._calibrate_group(group_logits, group)
		elif self.cfg["imax_cal_mode"] == "CW":
			for c in range(processed_logits.shape[1]):
				calibrated_probs[:, c] = self._calibrate_single_class(processed_logits[:, c], c)
		elif self.cfg["imax_cal_mode"] == "Top1":
			calibrated_probs = self._calibrate_top1(processed_logits)

		return calibrated_probs.numpy()

	def _run_em_optimization(self, logits: torch.Tensor, labels: torch.Tensor):
		prev_mi = -np.inf
		for iteration in range(self.cfg["em_iterations"]):
			# Unified EM flow
			if self.cfg["imax_cal_mode"] == "SCW":
				assignments = self._e_step_scw(logits)
				self._m_step_scw(logits, labels, assignments)
			else:
				assignments = self._e_step(logits, labels)
				self._m_step(logits, labels, assignments)

			# Track convergence
			current_mi = self._track_mutual_info(labels, assignments)
			if np.abs(current_mi - prev_mi) < 1e-5:
				break
			prev_mi = current_mi

	def _create_group_boundaries(self, group_logits: torch.Tensor) -> torch.Tensor:
		"""Create shared bin boundaries for a class group"""
		if self.cfg["bin_init_mode"] == "kmeans":
			return self._group_kmeans_init(group_logits)
		return self._group_quantile_init(group_logits)

	def _group_kmeans_init(self, logits: torch.Tensor) -> torch.Tensor:
		"""K-means initialization for group boundaries"""
		try:
			valid_logits = logits[~torch.isnan(logits)]
			probs = torch.softmax(valid_logits, dim=0).cpu().numpy()

			# Existing K-means logic adapted for groups
			n_samples = len(probs)
			if n_samples < self.cfg["imax_num_bins"]:
				raise ValueError(f"Not enough samples ({n_samples}) for clustering")

			# Vectorized JSD calculation
			p = np.vstack([probs, 1 - probs]).T
			q = p[:, np.newaxis, :]
			jsd_matrix = jensenshannon(p, q, axis=2) ** 2
			jsd_matrix = np.squeeze(jsd_matrix)

			# Single K-means fit with warning suppression
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=ConvergenceWarning)
				kmeans = KMeans(
					n_clusters=self.cfg["imax_num_bins"],
					init='k-means++',
					n_init=10,
					random_state=self.seed
				).fit(jsd_matrix)

			# Cluster validation
			unique_clusters = np.unique(kmeans.labels_)
			if len(unique_clusters) < self.cfg["imax_num_bins"]:
				raise ValueError(f"Only {len(unique_clusters)} clusters formed")

			# Handle empty clusters
			cluster_probs = []
			for i in range(self.cfg["imax_num_bins"]):
				mask = kmeans.labels_ == i
				if np.sum(mask) == 0:
					# Fallback to median of all samples
					cluster_probs.append(np.median(probs))
				else:
					cluster_probs.append(np.nanmean(probs[mask]))

			# Convert to logit space
			cluster_probs = np.clip(cluster_probs, 1e-10, 1 - 1e-10)
			cluster_centers = np.log(cluster_probs / (1 - cluster_probs))

			# Create boundaries
			sorted_centers = np.sort(cluster_centers)
			if len(sorted_centers) < 2:
				min_val = valid_logits.min().item()
				max_val = valid_logits.max().item()
				boundaries = np.linspace(min_val, max_val, self.cfg["imax_num_bins"] + 1)
			else:
				boundaries = np.zeros(self.cfg["imax_num_bins"] + 1)
				boundaries[0] = valid_logits.min().item() - 1e-6
				boundaries[-1] = valid_logits.max().item() + 1e-6
				boundaries[1:-1] = (sorted_centers[:-1] + sorted_centers[1:]) / 2

			# Final validation and conversion
			boundaries = np.unique(boundaries)
			if len(boundaries) < self.cfg["imax_num_bins"] + 1:
				boundaries = np.linspace(boundaries[0], boundaries[-1],
				                         self.cfg["imax_num_bins"] + 1)

			return torch.from_numpy(boundaries).float().contiguous()  # Critical fix!

		except Exception as e:
			print(f"Group K-means failed: {e}, using quantile")
			return self._group_quantile_init(logits)

	def _group_quantile_init(self, logits: torch.Tensor) -> torch.Tensor:
		"""Quantile initialization for group boundaries"""
		valid_logits = logits[~torch.isnan(logits)]
		sorted_logits = torch.sort(valid_logits).values
		return torch.quantile(
			sorted_logits,
			torch.linspace(0, 1, self.cfg["imax_num_bins"] + 1)
		).contiguous()
	def _e_step(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
		assignments = {}
		if self.cfg["imax_cal_mode"] == "CW":
			for c in range(logits.shape[1]):
				assignments[c] = torch.bucketize(
					logits[:, c].contiguous(),
					self.bin_boundaries[c].contiguous()
				).clamp(0, self.cfg["imax_num_bins"] - 1)

		elif self.cfg["imax_cal_mode"] == "Top1":
			top1_logits = logits[torch.arange(len(logits)), labels]
			assignments["top1"] = torch.bucketize(
				top1_logits.contiguous(),
				self.bin_boundaries[0].contiguous()
			).clamp(0, self.cfg["imax_num_bins"] - 1)

		return assignments

	def _m_step(self, logits: torch.Tensor, labels: torch.Tensor, assignments: dict):
		if self.cfg["imax_cal_mode"] == "CW":
			for c in range(logits.shape[1]):
				class_mask = labels == c
				self._update_class_params(
					logits[:, c],
					class_mask,
					assignments[c],
					c
				)
		elif self.cfg["imax_cal_mode"] == "Top1":
			top1_mask = torch.zeros_like(logits, dtype=torch.bool)
			top1_mask[torch.arange(len(logits)), labels] = True
			self._update_class_params(
				logits[top1_mask],
				torch.ones(len(logits), dtype=torch.bool),
				assignments["top1"],
				0
			)

		# Ensure boundaries remain sorted after updates
		for c in range(self.bin_boundaries.shape[0]):
			sorted_idx = torch.argsort(self.bin_boundaries[c])
			valid_bins = min(len(sorted_idx) - 1, self.cfg["imax_num_bins"])
			self.bin_boundaries[c] = self.bin_boundaries[c][sorted_idx[:valid_bins + 1]]
			# Create mask for valid indices
			valid_mask = sorted_idx < self.bin_probs.shape[1]
			sorted_probs = self.bin_probs[c][sorted_idx[valid_mask]]

			# Pad with default values if needed
			if len(sorted_probs) < self.cfg["imax_num_bins"]:
				pad_size = self.cfg["imax_num_bins"] - len(sorted_probs)
				sorted_probs = torch.cat([
					sorted_probs,
					torch.full((pad_size,), 0.5, dtype=sorted_probs.dtype)
				])

			self.bin_probs[c] = sorted_probs[:self.cfg["imax_num_bins"]]
	def _update_class_params(self, logits: torch.Tensor, labels: torch.Tensor,
	                         assignments: torch.Tensor, class_idx: int):
		for b in range(self.cfg["imax_num_bins"]):
			bin_mask = assignments == b
			n_bin_samples = bin_mask.sum().item()

			# Handle empty bins first
			if n_bin_samples == 0:
				self.bin_probs[class_idx][b] = 0.0  # or self.cfg["epsilon"]
				continue

			if n_bin_samples < self.cfg["min_bin_samples"]:
				# Merge with neighbor only if neighbor has samples
				valid_neighbors = []
				if b > 0 and (assignments == b - 1).sum() > 0:
					valid_neighbors.append(b - 1)
				if b < self.cfg["imax_num_bins"] - 1 and (assignments == b + 1).sum() > 0:
					valid_neighbors.append(b + 1)

				if not valid_neighbors:
					continue

				merge_bin = valid_neighbors[0]  # Prefer left neighbor
				merge_samples = (assignments == merge_bin).sum().item()

				# Safe weighted average
				current_prob = labels[bin_mask].float().mean().item()
				merged_prob = (
						              (self.bin_probs[class_idx][merge_bin] * merge_samples) +
						              (current_prob * n_bin_samples)
				              ) / (merge_samples + n_bin_samples)

				self.bin_probs[class_idx][merge_bin] = merged_prob
				continue

			# Regular update with clipping
			bin_prob = labels[bin_mask].float().mean()
			if torch.isnan(bin_prob):
				# Handle empty bins using neighbor average
				left_neighbor = max(0, b - 1)
				right_neighbor = min(self.cfg["imax_num_bins"] - 1, b + 1)
				bin_prob = (self.bin_probs[class_idx][left_neighbor] +
				            self.bin_probs[class_idx][right_neighbor]) / 2

			self.bin_probs[class_idx][b] = torch.clamp(
				bin_prob,
				self.cfg["epsilon"],
				1 - self.cfg["epsilon"]
			)

	def _update_group_params(self, group_logits: torch.Tensor,
	                         group_labels: torch.Tensor,
	                         assignments: torch.Tensor, group_key: tuple):
		"""Update parameters for a group in SCW mode"""
		# Ensure tensors have same dimensions
		assert len(group_logits) == len(group_labels) == len(assignments), \
			f"Dimension mismatch: {len(group_logits)} vs {len(group_labels)} vs {len(assignments)}"

		# Current parameters
		boundaries = self.group_boundaries[group_key]
		probs = self.group_probs[group_key]

		# 1. Update probabilities with merging logic
		new_probs = torch.zeros_like(probs)
		for b in range(self.cfg["imax_num_bins"]):
			bin_mask = assignments == b
			n_samples = bin_mask.sum().item()

			if n_samples == 0:
				new_probs[b] = probs[b]  # Maintain current value
				continue

			# Merge underpopulated bins
			if n_samples < self.cfg["min_bin_samples"]:
				# Find nearest non-empty neighbor
				left_neighbor = next(
					(i for i in reversed(range(b))
					 if (assignments == i).sum() > 0),
					None
				)
				right_neighbor = next(
					(i for i in range(b + 1, self.cfg["imax_num_bins"])
					 if (assignments == i).sum() > 0),
					None
				)

				# Prefer left neighbor if both exist
				merge_target = left_neighbor or right_neighbor

				if merge_target is not None:
					merged_samples = (assignments == merge_target).sum().item()
					total_samples = n_samples + merged_samples
					new_probs[merge_target] = (
							                          probs[merge_target] * merged_samples +
							                          group_labels[bin_mask].float().mean() * n_samples
					                          ) / total_samples
					continue

			# Regular update
			bin_prob = group_labels[bin_mask].float().mean()
			new_probs[b] = torch.clamp(
				torch.nan_to_num(bin_prob, nan=0.5),
				self.cfg["epsilon"],
				1 - self.cfg["epsilon"]
			)

		# 2. Update boundaries using closed-form solution
		new_boundaries = torch.zeros_like(boundaries)
		new_boundaries[0] = boundaries[0]
		new_boundaries[-1] = boundaries[-1]

		with torch.no_grad():
			for b in range(1, self.cfg["imax_num_bins"]):
				p_prev = new_probs[b - 1]
				p_curr = new_probs[b]

				# Numerical stability checks
				eps = self.cfg["epsilon"]
				p_prev = torch.clamp(p_prev, eps, 1 - eps)
				p_curr = torch.clamp(p_curr, eps, 1 - eps)

				# Compute boundary using paper's equation (6)
				numerator = torch.log((p_prev * (1 - p_curr)) / (p_curr * (1 - p_prev)))
				denominator = (1 / p_prev + 1 / (1 - p_prev) - 1 / p_curr - 1 / (1 - p_curr))

				if abs(denominator) < 1e-10:
					# Fallback to midpoint if denominator near zero
					new_boundaries[b] = (boundaries[b - 1] + boundaries[b]) / 2
				else:
					new_boundaries[b] = numerator / denominator

		# 3. Maintain sorted order
		sorted_indices = torch.argsort(new_boundaries)

		# Convert boundary indices to probability indices
		prob_indices = sorted_indices[:-1].clamp(max=len(new_probs) - 1)

		self.group_boundaries[group_key] = new_boundaries[sorted_indices]
		self.group_probs[group_key] = new_probs[prob_indices]

		# 4. Final clamping and validation
		self.group_probs[group_key] = torch.clamp(
			self.group_probs[group_key],
			self.cfg["epsilon"],
			1 - self.cfg["epsilon"]
		)
	def _validate_group_structure(self, group: tuple):
		"""Ensure group bin structure integrity"""
		boundaries = self.group_boundaries[group]
		probs = self.group_probs[group]

		# Check boundary count
		if len(boundaries) != self.cfg["imax_num_bins"] + 1:
			boundaries = torch.linspace(
				boundaries.min(),
				boundaries.max(),
				self.cfg["imax_num_bins"] + 1
			)

		# Ensure probabilities match bin count
		if len(probs) != self.cfg["imax_num_bins"]:
			probs = torch.full(
				(self.cfg["imax_num_bins"],), 0.5,
				dtype=probs.dtype
			)

		# Handle NaN values
		probs = torch.where(
			torch.isnan(probs),
			torch.full_like(probs, 0.5),
			probs
		)

		self.group_boundaries[group] = boundaries.contiguous()
		self.group_probs[group] = probs.clamp(
			self.cfg["epsilon"], 1 - self.cfg["epsilon"])
	def _validate_bin_structure(self, class_idx: int):
		"""Ensure bin structure integrity after initialization"""
		# Check boundary count
		if len(self.bin_boundaries[class_idx]) != self.cfg["imax_num_bins"] + 1:
			new_boundaries = torch.linspace(
				self.bin_boundaries[class_idx][0],
				self.bin_boundaries[class_idx][-1],
				self.cfg["imax_num_bins"] + 1
			)
			self.bin_boundaries[class_idx] = new_boundaries

		# Ensure boundaries are sorted and contiguous
		sorted_idx = torch.argsort(self.bin_boundaries[class_idx])
		self.bin_boundaries[class_idx] = self.bin_boundaries[class_idx][sorted_idx].contiguous()

		# Initialize probabilities if needed
		if torch.any(torch.isnan(self.bin_probs[class_idx])):
			self.bin_probs[class_idx] = torch.full_like(self.bin_probs[class_idx], 0.5)

	def _quantile_init(self, logits: torch.Tensor, class_idx: int):
		"""Robust quantile initialization with uniform distribution handling"""
		sorted_logits = torch.sort(logits).values
		if torch.allclose(sorted_logits, sorted_logits[0].expand_as(sorted_logits), atol=1e-6):
			# Handle uniform distribution case
			min_val = sorted_logits[0].item() - 1e-3
			max_val = sorted_logits[0].item() + 1e-3
			boundaries = torch.linspace(
				min_val, max_val,
				self.cfg["imax_num_bins"] + 1
			)
		else:
			try:
				boundaries = torch.quantile(
					sorted_logits,
					torch.linspace(0, 1, self.cfg["imax_num_bins"] + 1)
				)
			except RuntimeError:
				boundaries = torch.linspace(
					sorted_logits.min(), sorted_logits.max(),
					self.cfg["imax_num_bins"] + 1
				)

		self.bin_boundaries[class_idx] = boundaries.contiguous()
	def _track_mutual_info(self, labels: torch.Tensor, assignments: dict) -> float:
		if self.cfg["imax_cal_mode"] == "SCW":
			mi_values = []
			for group in self.class_groups:
				group_key = tuple(group)
				group_labels = torch.isin(labels, torch.tensor(group)).int().numpy()  # Shape (N,)

				group_assignments = assignments[group_key].numpy()  # Shape (N,)

				# Verify dimensions match
				assert len(group_labels) == len(group_assignments), \
					f"SCW MI dimension mismatch: {len(group_labels)} vs {len(group_assignments)}"

				mi = adjusted_mutual_info_score(group_labels, group_assignments)
				mi_values.append(mi)
			current_mi = np.mean(mi_values)
		elif self.cfg["imax_cal_mode"] == "CW":
			all_assignments = torch.cat([a.flatten() for a in assignments.values()])
			all_labels = torch.cat([(labels == c).int() for c in assignments.keys()])
			current_mi = mutual_info_score(all_labels.numpy(), all_assignments.numpy())
		elif self.cfg["imax_cal_mode"] == "Top1":
			current_mi = mutual_info_score(
				torch.ones(len(assignments["top1"])).numpy(),
				assignments["top1"].numpy()
			)

		self.mi_history.append(current_mi)
		return current_mi

	def _e_step_scw(self, logits: torch.Tensor) -> dict:
		assignments = {}
		for group in self.class_groups:
			group_key = tuple(group)
			group_logits_agg = torch.logsumexp(logits[:, group], dim=1)  # (N,)
			assignments[group_key] = torch.bucketize(
				group_logits_agg,
				self.group_boundaries[group_key]
			)
		return assignments

	def _m_step_scw(self, logits: torch.Tensor, labels: torch.Tensor, assignments: dict):
		"""M-step for SCW mode updating group parameters"""
		for group in self.class_groups:
			group_key = tuple(group)

			# Get pre-computed group assignments from E-step
			group_assignments = assignments[group_key]  # Already (N,) from fixed E-step

			# Aggregate group logits (now matching assignments)
			group_logits_agg = torch.logsumexp(logits[:, group], dim=1)  # (N,)

			# Get group membership labels
			group_labels = torch.isin(labels, torch.tensor(group)).float()  # (N,)

			# Update parameters with aligned dimensions
			self._update_group_params(
				group_logits_agg,
				group_labels,
				group_assignments,
				group_key
			)
	def _initialize_bins(self, logits: torch.Tensor):
		if self.cfg["imax_cal_mode"] == "SCW":
			# SCW initialization
			self.group_boundaries = {}
			self.group_probs = {}
			for group in self.class_groups:
				group_logits = torch.cat([logits[:, c] for c in group])
				self.group_boundaries[tuple(group)] = self._create_group_boundaries(group_logits)
				self.group_probs[tuple(group)] = torch.full(
					(self.cfg["imax_num_bins"],), 0.5
				)
		else:
			for c in range(logits.shape[1]):
				class_logits = logits[:, c]
				valid_logits = class_logits[~torch.isnan(class_logits)]

				if self.cfg["bin_init_mode"] == "kmeans":
					try:
						# Convert to probabilities with numerical stability
						probs = torch.softmax(valid_logits, dim=0)
						probs = torch.clamp(probs, 1e-10, 1 - 1e-10).cpu().numpy()

						# Fallback check for uniform probabilities
						if np.allclose(probs, probs[0], atol=1e-6):
							raise ValueError("Uniform probabilities detected")

						# Compute pairwise JSD distances with vectorization
						n_samples = len(probs)
						jsd_matrix = np.zeros((n_samples, n_samples))

						# Vectorized JSD calculation
						p = np.vstack([probs, 1 - probs]).T
						q = p[:, np.newaxis, :]
						jsd_matrix = jensenshannon(p, q, axis=2) ** 2
						jsd_matrix = np.squeeze(jsd_matrix)

						# KMeans with cluster validation
						kmeans = KMeans(
							n_clusters=self.cfg["imax_num_bins"],
							init='k-means++',
							n_init=10,
							random_state=self.seed
						)

						with warnings.catch_warnings():
							warnings.filterwarnings("ignore", category=ConvergenceWarning)
							kmeans.fit(jsd_matrix)

						# Cluster validation and fallback
						unique_clusters = np.unique(kmeans.labels_)
						if len(unique_clusters) < self.cfg["imax_num_bins"]:
							raise ValueError(f"Only {len(unique_clusters)} clusters formed")

						# Handle empty clusters robustly
						cluster_probs = []
						for i in range(self.cfg["imax_num_bins"]):
							mask = kmeans.labels_ == i
							if np.sum(mask) == 0:
								# Fallback to nearest non-empty cluster
								non_empty = [j for j in range(self.cfg["imax_num_bins"]) if
								             j != i and np.sum(kmeans.labels_ == j) > 0]
								if non_empty:
									cluster_probs.append(np.median(probs[kmeans.labels_ == non_empty[0]]))
								else:
									cluster_probs.append(np.median(probs))
							else:
								cluster_probs.append(np.nanmean(probs[mask]))

						# Convert back to logit space with stability
						cluster_probs = np.array(cluster_probs)
						cluster_probs = np.clip(cluster_probs, 1e-10, 1 - 1e-10)
						cluster_centers = np.log(cluster_probs / (1 - cluster_probs))

						# Create boundaries with padding
						sorted_centers = np.sort(cluster_centers)
						if len(sorted_centers) < 2:
							min_val = valid_logits.min().item() - 1e-3
							max_val = valid_logits.max().item() + 1e-3
							boundaries = np.linspace(min_val, max_val, self.cfg["imax_num_bins"] + 1)
						else:
							boundaries = np.zeros(self.cfg["imax_num_bins"] + 1)
							boundaries[0] = valid_logits.min().item() - 1e-6
							boundaries[-1] = valid_logits.max().item() + 1e-6
							boundaries[1:-1] = (sorted_centers[:-1] + sorted_centers[1:]) / 2

						# Ensure boundaries are unique and sorted
						boundaries = np.unique(boundaries)
						if len(boundaries) < self.cfg["imax_num_bins"] + 1:
							boundaries = np.linspace(
								boundaries[0], boundaries[-1],
								self.cfg["imax_num_bins"] + 1
							)

						self.bin_boundaries[c] = torch.from_numpy(boundaries).float().contiguous()

					except Exception as e:
						print(f"JSD-KMeans failed ({str(e)}), falling back to quantile")
						self._quantile_init(valid_logits, c)

				elif self.cfg["bin_init_mode"] == "quantile":
					self._quantile_init(valid_logits, c)

				# Post-initialization validation
				self._validate_bin_structure(c)
	def _form_class_groups(self):
		if self.cfg["imax_cal_mode"] != "SCW":
			return

		# Sort classes by prior probabilities
		sorted_indices = torch.argsort(self.class_priors, descending=True)
		sorted_classes = sorted_indices.tolist()

		# Form groups with similar priors
		self.class_groups = []
		current_group = []
		current_prior = None

		for c in sorted_classes:
			if self.class_priors[c] < 1e-6:
				continue

			if not current_group:
				current_group.append(c)
				current_prior = self.class_priors[c]
			else:
				if abs(self.class_priors[c] - current_prior) < self.cfg["merge_threshold"]:
					current_group.append(c)
				else:
					self.class_groups.append(current_group)
					current_group = [c]
					current_prior = self.class_priors[c]

		if current_group:
			self.class_groups.append(current_group)

	def _compute_class_priors(self, labels: torch.Tensor):
		counts = torch.bincount(labels, minlength=len(self.class_priors))
		self.class_priors = counts.float() / counts.sum()

	def _preprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
		if self.cfg["Q_binning_stage"] == "scaled":
			return logits / self.cfg["temperature"]
		return logits

	def _calibrate_group(self, group_logits: torch.Tensor, group: list) -> torch.Tensor:
		"""Fixed SCW calibration using group parameters"""
		calibrated = torch.zeros_like(group_logits)
		group_key = tuple(group)
		for g, c in enumerate(group):
			# Use group boundaries instead of class boundaries
			bin_indices = torch.bucketize(
				group_logits[:, g],
				self.group_boundaries[group_key],
				right=True
			).clamp(0, self.cfg["imax_num_bins"] - 1)

			# Use group probabilities
			calibrated[:, g] = self.group_probs[group_key][bin_indices]
		return calibrated

	def _calibrate_single_class(self, logits: torch.Tensor, class_idx: int) -> torch.Tensor:
		bin_indices = torch.bucketize(
			logits,
			self.bin_boundaries[class_idx],
			right=True
		).clamp(0, self.cfg["imax_num_bins"] - 1)
		return self.bin_probs[class_idx][bin_indices]

	def _calibrate_top1(self, logits: torch.Tensor) -> torch.Tensor:
		top1_probs = torch.zeros_like(logits)
		max_indices = torch.argmax(logits, dim=1)
		top1_values = logits[torch.arange(len(logits)), max_indices]

		bin_indices = torch.bucketize(
			top1_values,
			self.bin_boundaries[0],
			right=True
		).clamp(0, self.cfg["imax_num_bins"] - 1)

		calibrated_values = self.bin_probs[0][bin_indices]
		top1_probs[torch.arange(len(logits)), max_indices] = calibrated_values
		return top1_probs

	def get_calibration_parameters(self):
		return {
			"bin_boundaries": self.bin_boundaries.numpy(),
			"bin_probs": self.bin_probs.numpy(),
			"class_groups": self.class_groups,
			"mi_history": self.mi_history
		}

	def get_timing(self):
		"""Return the timing information for fit and predict operations."""
		return self.timing