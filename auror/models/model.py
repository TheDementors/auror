import torch


class Model:
    @staticmethod
    def __detect_hardware(training_device, force_cpu):
        """Private method for the Model class that automatically detects the hardware
        to use (CPU or CUDA) based on the passed parameters on the constructor.

        Args:
            training_device (torch.device | None): torch device or None
            force_cpu (bool): When to use CPU under any circusstance
                even if GPU is available

        Returns:
            (torch.device): returns a torch device to be used for training
        """
        if force_cpu:
            return torch.device("cpu")

        if training_device:
            return training_device

        if torch.cuda.is_available():
            # TODO: Do something if there is more than 1 GPU
            return torch.device("cuda:0")

        return torch.device("cpu")

    def __init__(self, training_device=None, force_cpu=False):
        """Auror Model class

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
        if not (isinstance(training_device, torch.device) or training_device is None):
            raise ValueError("Please provide a valid training_device parameter")

        # Checking the force_cpu parameter
        if not isinstance(force_cpu, bool):
            raise ValueError(
                "You have provided an invalid value for the parameter force_cpu"
            )

        # Detecting the hardware to train the model
        self.__device = self.__detect_hardware(training_device, force_cpu)
        print(f"--> Using {str(self.__device).upper()} as a traning device")

    def attach_pytorch_model(self, model):
        """Attaches a PyTorch model to the Auror Model class

        Args:
            model (pytorch model): A Sequential and Class based PyTorch model

        Raises:
            ValueError: If it receives an invalid PyTorch model
        """
        # Checking the model parameter
        # TODO: Need to check the model is valid PyTorch model
        if model is None:
            raise ValueError("Please provide a class based PyTorch model")

        self.__model = model

    def attach_loss_function(self, loss_function):
        """Attaches a PyTorch loss function to the Auror Model class

        Args:
            loss_function (pytorch loss function): A valid PyTorch loss function

        Raises:
            ValueError: If it receives an invalid PyTorch loss function
        """
        # Checking the loss_function parameter
        # TODO: Need to check the loss_function is valid PyTorch loss_function
        if loss_function is None:
            raise ValueError("Please provide a valid PyTorch loss function")

        self.__loss_function = loss_function

    def attach_optimizer(self, optimizer):
        """Attaches a PyTorch optimizer to the Auror Model class

        Args:
            model (pytorch optimizer): A valid PyTorch optimizer

        Raises:
            ValueError: If it receives an invalid PyTorch optimizer
        """
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
        """Prints a summary of the model with all the layers, number of
        Trainable and Non-Trainable parameters.

        Raises:
            ValueError: If there is no model
        """
        if not self.__model:
            raise ValueError("There is no compiled model to generate a summary")

        print(self.__model)
        print(
            "Total Number \
            of Parameters: ",
            sum(p.numel() for p in self.__model.parameters()),
        )
        print(
            "Total Number of Trainable \
            Parameters: ",
            sum(p.numel() for p in self.__model.parameters() if p.requires_grad),
        )

    def save(self, path):
        """The .save() method is responsible for saving a trained model. This method is
        to be shared with someone without any code access.

        Args:
            path (string): The path to store the data

        Raises:
            ValueError: If there is no correct path passed to this method
        """
        if not path or not isinstance(path, str):
            raise ValueError("Please provide a valid path")

        torch.save(self.__model, path)

    def load(self, path):
        """The .load() method is responsible for loading a model saved using the
        .save() method.

        Args:
            path (string): Path of the saved Auror model

        Raises:
            ValueError: If the path is not valid
        """
        if not path or not isinstance(path, str):
            raise ValueError("Please provide a valid path")

        self.__model = torch.load(path)

    def save_for_inference(self, path):
        """The .save_for_inference() method is responsible for saving a trained model.
        This method saves the method only for inference.

        Args:
            path (string): Path to save the model

        Raises:
            ValueError: If the path is not valid
        """
        if not path or not isinstance(path, str):
            raise ValueError("Please provide a valid path")

        torch.save(self.__model.state_dict(), path)

    def load_for_inference(self, path):
        """The .load_for_inference() method is responsible for loading a trained model
        only for inference.

        Args:
            path (string): Path where the model is saved

        Raises:
            ValueError: If the path is not valid or the if there is valid model loaded
                in the class
        """
        if not path or not isinstance(path, str):
            raise ValueError("Please provide a valid path")

        if self.__model:
            self.__model.load_state_dict(torch.load(path))
        else:
            raise ValueError("To load the model state, you need to have a model first")
