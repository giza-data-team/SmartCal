import unittest
import os
import tempfile
import torch
import numpy as np
import numpy.testing as npt
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FakeData
from torchvision.transforms import Compose, Resize, ToTensor

from classifiers.image_classifier import ImageClassifier
from smartcal.config.enums.image_models_enum import ImageModelsEnum


class EmptyDataset(Dataset):
    """A dataset that contains no samples."""
    def __len__(self):
        return 0  # No samples

    def __getitem__(self, idx):
        return torch.zeros((3, 224, 224)), 0  # Dummy image and label

class TestImageClassifier(unittest.TestCase):
    """Comprehensive unit tests for the ImageClassifier class."""

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cls.SEED = 42

        torch.manual_seed(cls.SEED)
        if cls.DEVICE == "cuda":
            torch.cuda.manual_seed_all(cls.SEED)

        cls.temp_dir = tempfile.mkdtemp()

        # Create datasets for both training and validation
        transform = Compose([Resize((224, 224)), ToTensor()])
        train_dataset = FakeData(transform=transform, size=100)
        val_dataset = FakeData(transform=transform, size=20) 
        
        cls.dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        cls.val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        cls.models = [
            ImageModelsEnum.RESNET18,
            #ImageModelsEnum.MOBILENET,
            #ImageModelsEnum.VGG16,
            #ImageModelsEnum.VGG19,
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up shared resources after all tests."""
        if os.path.exists(cls.temp_dir):
            for file in os.listdir(cls.temp_dir):
                os.remove(os.path.join(cls.temp_dir, file))
            os.rmdir(cls.temp_dir)

    def test_train(self):
        """Test model training."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
                self.assertGreaterEqual(classifier.training_metrics["accuracy"], 0)
                self.assertLessEqual(classifier.training_metrics["accuracy"], 100)

    def test_invalid_dataloader(self):
        """Ensure that passing an invalid dataloader type raises a TypeError."""
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        with self.assertRaises(TypeError):  
            classifier.train("invalid_dataloader", epochs=1, num_itr=2)

        with self.assertRaises(TypeError):  
            classifier.predict("invalid_dataloader")

    def test_invalid_num_classes(self):
        """Ensure that invalid `num_classes` values raise an error."""
        with self.assertRaises(RuntimeError):
            ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=-1, device=self.DEVICE)

        with self.assertRaises(RuntimeError):
            ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=0, device=self.DEVICE)

        with self.assertRaises(RuntimeError):
            ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes="invalid", device=self.DEVICE)

    def test_training_on_empty_dataset(self):
        """Ensure that training on an empty dataset raises an error."""
        empty_dataloader = DataLoader(EmptyDataset(), batch_size=32)
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        with self.assertRaises(RuntimeError):
            classifier.train(empty_dataloader,empty_dataloader, epochs=1, num_itr=2)

    def test_training_on_single_sample(self):
        """Test training with only one image."""
        transform = Compose([Resize((224, 224)), ToTensor()])
        single_sample_dataset = FakeData(transform=transform, size=1)
        single_sample_dataloader = DataLoader(single_sample_dataset, batch_size=1)

        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        try:
            classifier.train(single_sample_dataloader,single_sample_dataloader, epochs=1, num_itr=2)
        except Exception as e:
            self.fail(f"Training on a single sample should not fail. Error: {e}")

    def test_training_multiple_epochs(self):
        """Ensure training over multiple epochs updates the model."""
        classifier = ImageClassifier(
            model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
        )

        classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=3, num_itr=2)
        self.assertGreaterEqual(classifier.training_metrics["accuracy"], 0, "Accuracy should be valid after training.")

    def test_training_with_different_batch_sizes(self):
        """Ensure training works with different batch sizes."""
        for batch_size in [1, 16, 64]:
            dataloader = DataLoader(FakeData(transform=Compose([Resize((224, 224)), ToTensor()]), size=100), batch_size=batch_size, shuffle=False)
            classifier = ImageClassifier(
                model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
            )
            classifier.train(dataloader=dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
            self.assertGreaterEqual(classifier.training_metrics["accuracy"], 0, "Accuracy should be valid after training.")

    def test_training_without_labels(self):
        """Ensure that training without labels raises an error."""
        class NoLabelsDataset(Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return torch.randn((3, 224, 224))  # No label provided

        dataloader = DataLoader(NoLabelsDataset(), batch_size=2)
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        with self.assertRaises(ValueError):  # Expecting ValueError instead of RuntimeError
            classifier.train(dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

    def test_training_on_single_class(self):
        """Ensure model trains correctly when all samples belong to one class."""
        class SingleClassDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return torch.randn((3, 224, 224)), torch.tensor(0)  # Always class 0

        dataloader = DataLoader(SingleClassDataset(), batch_size=8)
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        classifier.train(dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
        self.assertGreaterEqual(classifier.training_metrics["accuracy"], 0, "Accuracy should be valid after training.")

    def test_predict(self):
        """Test model prediction."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
                predictions = classifier.predict(self.dataloader)
                self.assertEqual(len(predictions), len(self.dataloader.dataset))

    def test_predict_single_image(self):
        """Ensure model can handle inference on a single image."""
        transform = Compose([Resize((224, 224)), ToTensor()])
        single_image_dataset = FakeData(transform=transform, size=1)
        single_image_dataloader = DataLoader(single_image_dataset, batch_size=1)

        classifier = ImageClassifier(
            model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
        )
        classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

        predictions = classifier.predict(single_image_dataloader)
        self.assertEqual(len(predictions), 1, "Should return a prediction for the single image.")

    def test_predict_with_different_image_sizes(self):
        """Ensure model can handle different image sizes correctly."""
        class VariableSizeDataset(Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                size = 224 if idx % 2 == 0 else 256  # Alternate between 224x224 and 256x256
                return torch.randn((3, size, size)), torch.randint(0, 10, (1,))

        dataloader = DataLoader(VariableSizeDataset(), batch_size=2)
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)
        
        classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)  # Train first

        with self.assertRaises(RuntimeError):
            classifier.predict(dataloader)

    def test_predict_with_corrupt_images(self):
        """Ensure prediction handles corrupt images correctly."""
        class CorruptDataset(Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                if idx % 2 == 0:
                    return torch.randn((3, 224, 224)), torch.randint(0, 10, (1,))
                else:
                    return None, None  # Corrupt image

        dataloader = DataLoader(CorruptDataset(), batch_size=2)
        classifier = ImageClassifier(model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE)

        classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)  # Train first

        with self.assertRaises(RuntimeError):
            classifier.predict(dataloader)

    def test_model_evaluation_without_training(self):
        """Ensure that evaluating an untrained model does not crash."""
        classifier = ImageClassifier(
            model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
        )

        with self.assertRaises(RuntimeError):
            classifier.predict(self.dataloader)

    def test_predict_prob(self):
        """Test model probability prediction."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
                predictions = classifier.predict(self.dataloader)
                probabilities = classifier.predict_prob(self.dataloader)
                self.assertEqual(len(probabilities), len(self.dataloader.dataset))

    def test_predict_prob_list_sizes(self):
        """Ensure predict_prob() returns lists of correct sizes matching batch sizes."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,  # Ensure num_classes is properly set
                    device=self.DEVICE,
                    seed=self.SEED
                )

                # Train the model
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

                # Predict Probabilities
                predictions = classifier.predict(self.dataloader)
                probabilities = classifier.predict_prob(self.dataloader)

                # Ensure probabilities is a list of lists
                self.assertIsInstance(probabilities, np.ndarray, "Output of predict_prob() should be a list.")
                self.assertIsInstance(probabilities[0], np.ndarray, "Each entry in predict_prob() should be a list.")

                # Ensure the number of probability lists matches the dataset size
                self.assertEqual(len(probabilities), len(self.dataloader.dataset),
                                "The number of probability lists should match the number of samples.")

                # Ensure each probability list has the correct number of classes
                num_classes = classifier.model.fc.out_features  # Dynamically get output layer size
                for prob in probabilities:
                    self.assertEqual(len(prob), num_classes,
                                    "Each probability list should have an entry for every class.")

    def test_predict_prob_on_untrained_model(self):
        """Ensure calling `predict_prob()` on an untrained model does not crash."""
        classifier = ImageClassifier(
            model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
        )

        with self.assertRaises(RuntimeError):
            predictions = classifier.predict(self.dataloader)
            classifier.predict_prob(self.dataloader)

    def test_save(self):
        """Test model saving."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)
                model_path = os.path.join(self.temp_dir, f"{model_enum.model_name}_classifier.pth")
                classifier.save_model(model_path)
                self.assertTrue(os.path.exists(model_path))

    def test_save_invalid_path(self):
        """Ensure saving the model to an invalid file path raises an error."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

                # Attempt to save the model to an invalid path
                invalid_path = "/invalid/path/model.pth"
                with self.assertRaises(RuntimeError):
                    classifier.save_model(invalid_path)

    def test_save_permission_error(self):
        """Ensure saving to a restricted directory raises a permission error."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

                # Try saving to a restricted location (e.g., root directory)
                restricted_path = "/root/restricted_model.pth"
                with self.assertRaises(RuntimeError):
                    classifier.save_model(restricted_path)

    def test_save_no_space(self):
        """Simulate a disk space issue when saving a model."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

                # Mock `torch.save()` to raise an OSError (simulating disk full error)
                with unittest.mock.patch("torch.save", side_effect=OSError("No space left on device")):
                    with self.assertRaises(RuntimeError):
                        classifier.save_model(os.path.join(self.temp_dir, "full_disk.pth"))

    def test_load(self):
        """Test model loading when only model state is restored (no optimizer or training metrics)."""
        for model_enum in self.models:
            with self.subTest(model_enum=model_enum):
                classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )
                classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

                # Save the trained model
                model_path = os.path.join(self.temp_dir, f"{model_enum.model_name}_classifier.pth")
                classifier.save_model(model_path)

                # Create a new classifier instance
                new_classifier = ImageClassifier(
                    model_enum=model_enum,
                    num_classes=10,
                    device=self.DEVICE,
                    seed=self.SEED
                )

                # Load the model
                new_classifier.load_model(model_path)

                # Ensure model is in evaluation mode
                new_classifier.model.eval()

                # Manually mark model as "trained" to allow predictions
                new_classifier.training_metrics = classifier.training_metrics 

                # Verify that model parameters were restored correctly
                for param_original, param_loaded in zip(classifier.model.parameters(), new_classifier.model.parameters()):
                    self.assertTrue(torch.equal(param_original, param_loaded),
                                    "Model parameters are not the same after loading.")

                # Ensure model can make predictions after loading
                try:
                    predictions_after_loading = new_classifier.predict(self.dataloader)
                except RuntimeError:
                    self.fail("Model should be able to make predictions after loading.")

                # Ensure predictions remain consistent before and after saving/loading
                npt.assert_array_equal(
                    classifier.predict(self.dataloader),
                    predictions_after_loading,
                    "Predictions should remain the same after saving and loading."
                )

    def test_log_results(self):
        """Test the log_results function to ensure it returns the expected structure."""
        classifier = ImageClassifier(
            model_enum=ImageModelsEnum.RESNET18, num_classes=10, device=self.DEVICE
        )

        # Train the model
        classifier.train(dataloader=self.dataloader, val_loader=self.val_dataloader, epochs=1, num_itr=2)

        # Run predictions
        classifier.predict(self.dataloader)
        classifier.predict_prob(self.dataloader)

        # Get log results (should return a dictionary)
        log_data = classifier.log_results()

        # Ensure log_data is a dictionary
        self.assertIsInstance(log_data, dict, "log_results() should return a dictionary.")

        # Ensure expected keys exist
        expected_keys = [
            "model", "dataset_type", "device", "training_metrics",
            "testing_metrics", "training_time", "testing_time_predict", "testing_time_predictprob"
        ]

        for key in expected_keys:
            self.assertIn(key, log_data, f"Missing key `{key}` in log results.")

        # Ensure model name is correctly recorded
        self.assertEqual(log_data["model"], classifier.model.__class__.__name__)

        # Ensure training and testing metrics exist
        self.assertIsInstance(log_data["training_metrics"], dict, "Training metrics should be a dictionary.")
        self.assertIsInstance(log_data["testing_metrics"], dict, "Testing metrics should be a dictionary.")

        # Ensure accuracy exists in training/testing metrics
        self.assertIn("accuracy", log_data["training_metrics"], "Training metrics should contain `accuracy`.")
        self.assertIn("accuracy", log_data["testing_metrics"], "Testing metrics should contain `accuracy`.")

        # Ensure training and testing times are recorded
        self.assertIsInstance(log_data["training_time"], (int, float), "Training time should be a number.")
        self.assertGreaterEqual(log_data["training_time"], 0, "Training time should be non-negative.")

        self.assertIsInstance(log_data["testing_time_predict"], (int, float), "Testing time should be a number.")
        self.assertGreaterEqual(log_data["testing_time_predict"], 0, "Testing time should be non-negative.")

        self.assertIsInstance(log_data["testing_time_predictprob"], (int, float), "Testing time should be a number.")
        self.assertGreaterEqual(log_data["testing_time_predictprob"], 0, "Testing time should be non-negative.")

if __name__ == "__main__":
    unittest.main()
