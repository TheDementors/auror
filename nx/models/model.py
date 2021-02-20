from torch import device
from torch import cuda


class Model:
    def __init__(self, training_device=None, force_cpu=False):
        """NeuralPyX Model class

        This is class is used for building a NeuralPyX model from a PyTorch model.
        It only supports class based and sequential based PyTorch model.


        Args:
            training_device (torch.device, optional): Traning device
                for the PyTorch model. Defaults to None.
            force_cpu (bool, optional): When True then it always uses
                CPU even if CUDA is present. Defaults to False.

        Raises:
            ValueError: Please provide a valid training_device parameter
            ValueError: You have provided an invalid value for the parameter force_cpu
        """
        # Checking the training_device parameter
        if not (isinstance(training_device, device) or training_device is None):
            raise ValueError("Please provide a valid training_device parameter")

        # Checking the force_cpu parameter
        if not isinstance(force_cpu, bool):
            raise ValueError(
                "You have provided an invalid value for the parameter force_cpu"
            )

        # Detecting the hardware to train the model
        self.__detect_hardware(training_device, force_cpu)

    def __detect_hardware(self, training_device, force_cpu):
        if training_device:
            self.__device = training_device
        elif force_cpu:
            self.__device = device("cpu")
        else:
            if cuda.is_available():
                # TODO: Do something id there is more than 1 GPU
                self.__device = device("cuda:0")
            else:
                self.__device = device("cpu")

    def attach_pytorch_model(self, model):
        # Checking the model parameter
        # TODO: Need to check the model is valid PyTorch model
        if model is None:
            raise ValueError("Please provide a class based PyTorch model")

        self.__model = model

    def attach_loss_function(self, loss_function):
        # Checking the loss_function parameter
        # TODO: Need to check the loss_function is valid PyTorch loss_function
        if loss_function is None:
            raise ValueError("Please provide a valid PyTorch loss function")

        self.__loss_function = loss_function

    def attach_optimizer(self, optimizer):
        # Checking the optimizer parameter
        # TODO: Need to check the optimizer is valid PyTorch optimizer
        if optimizer is None:
            raise ValueError("Please provide a valid PyTorch optimizer")

        self.__optimizer = optimizer

    def fit(self, train_data, validation_data, batch_size):
        pass

    def predict(self, predict_data):
        pass

    def predict_classes(self, predict_data):
        pass

    def evaluate(self, evaluate_data):
        pass

    def summary(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def save_for_inference(self):
        pass

    def load_for_inference(self):
        pass
