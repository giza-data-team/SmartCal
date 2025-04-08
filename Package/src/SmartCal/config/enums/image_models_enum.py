from torchvision.models import mobilenet_v2, vgg16, vgg19, resnet18

from enum import Enum


class ImageModelsEnum(Enum):
    """
    Enum class for representing different image classification model architectures.
    Provides an easy way to select and use pretrained models from the torchvision library.
    """
    MOBILENET = ("MobileNetV2", mobilenet_v2)  # MobileNetV2 architecture
    VGG16 = ("VGG16", vgg16)  # VGG16 architecture
    VGG19 = ("VGG19", vgg19)  # VGG19 architecture
    RESNET18 = ("ResNet18", resnet18)  # ResNet18 architecture

    def __call__(self, *args, **kwargs):
        """
        Make the enum callable to directly call the model function.

        Args:
            *args: Variable positional arguments to pass to the model function.
            **kwargs: Variable keyword arguments to pass to the model function.

        Returns:
            A PyTorch model instance initialized with the provided arguments.
        """ 
        return self.value[1](*args, **kwargs)  # Call the model function

    @property
    def model_name(self):
        """
        Retrieve the model's name.

        Returns:
            str: The name of the model.
        """
        return self.value[0]
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
