import torch
import numpy as np

from py_progress import progressbar
from torchsummary import summary


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

    def __print_progress(self, current_batch, total_batches, epoch, epochs, loss, accuracy, validation=False):
        type = "Validation" if validation else "Training"

        left = f"{type} -> Epoch: {epoch+1}/{epochs} Batch: {current_batch+1}/{total_batches}"
        right = f"Loss: {loss:.4f} Accuracy: {accuracy:.2f}" if self.__is_classification else f"Loss: {loss:.4f}"

        progressbar(current_batch, total_batches, left, right)

    def __calculate_accuracy(y, y_pred):
        pred = y_pred.argmax(dim=1, keepdim=True)

        corrects = pred.eq(y.view_as(pred)).sum().item()

        return corrects

    def __model_loop(self, batch_x, batch_y, is_training):
        if len(batch_x) != len(batch_y):
            # length of X and y should be same
            raise ValueError("Length of training input data and training output data should be same")

        batch_x, batch_y = batch_x.to(self.__device), batch_y.to(self.__device)

        batch_x = batch_x.float()
        batch_y = batch_y.float()

        loss_score = 0
        corrects = 0

        if is_training:
            self.__model.zero_grad()

            outputs = self.__model(batch_x)
            loss = self.__loss_function(batch_y, outputs)

            loss.backward()
            self.__optimizer.step()

            loss_score += loss.item()

            if self.__is_classification:
                corrects += Model.__calculate_accuracy(batch_y, outputs)
        else:
            self.__model.eval()

            with torch.no_grad():
                outputs = self.__model(batch_x)
                loss = self.__loss_function(batch_y, outputs)

                loss_score += loss.item()

                if self.__is_classification:
                    corrects += Model.__calculate_accuracy(batch_y, outputs)

        return loss_score, corrects / len(batch_x) * 100

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
            raise ValueError("You have provided an invalid value for the parameter force_cpu")

        # Detecting the hardware to train the model
        self.__device = self.__detect_hardware(training_device, force_cpu)
        print(f"--> Using {str(self.__device).upper()} as a traning device")

    def parameters(self):
        return self.__model.parameters()

    def initialize_model(self, model):
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

    def compile(self, optimizer, loss_function, is_classification=True):
        if optimizer is None:
            raise ValueError("Please provide a valid PyTorch optimizer")

        if loss_function is None:
            raise ValueError("Please provide a valid PyTorch loss function")

        if not isinstance(is_classification, bool):
            raise ValueError("Please provide a valid value for the parameter is_classification")

        self.__optimizer = optimizer
        self.__loss_function = loss_function
        self.__is_classification = is_classification

    def fit(
        self,
        train_data,
        validation_data,
        batch_size=None,
        epochs=10,
        total_training_batches=None,
        total_validation_batches=None,
    ):
        for epoch in range(epochs):
            # Training
            if isinstance(train_data, tuple):
                batches, labels = train_data

                if not (isinstance(batches, np.ndarray) or isinstance(labels, np.ndarray)):
                    raise ValueError("Please provide a valid data to train")

                batches, labels = torch.from_numpy(batches), torch.from_numpy(labels)

                total_loss = 0
                total_corrects = 0

                for i in range(0, len(batches), batch_size):
                    batch_x = batches[i : i + batch_size]
                    batch_y = batches[i : i + batch_size]

                    total_loss, total_corrects = self.__model_loop(batch_x=batch_x, batch_y=batch_y, is_training=True)

                    total_batches = int(len(batches) / batch_size)
                    current_batch = int(i / batch_size)

                    self.__print_progress(current_batch, total_batches, epoch, epochs, total_loss, total_corrects)

                if validation_data:
                    print("")

                    val_batches, val_labels = validation_data

                    if not (isinstance(val_batches, np.ndarray) or isinstance(val_labels, np.ndarray)):
                        raise ValueError("Please provide a valid data to validate")

                    val_batches, val_labels = (torch.from_numpy(val_batches), torch.from_numpy(val_labels))

                    total_validation_loss = 0
                    total_validation_corrects = 0

                    for i in range(0, len(val_batches), batch_size):
                        batch_x = val_batches[i : i + batch_size]
                        batch_y = val_batches[i : i + batch_size]

                        (total_validation_loss, total_validation_corrects) = self.__model_loop(
                            batch_x=batch_x, batch_y=batch_y, is_training=False
                        )

                        total_batches = int(len(val_batches) / batch_size)
                        current_batch = int(i / batch_size)

                        self.__print_progress(
                            current_batch,
                            total_batches,
                            epoch,
                            epochs,
                            total_validation_loss,
                            total_validation_corrects,
                            validation=True,
                        )

                print("")
            else:
                if not isinstance(total_validation_batches, int) or total_validation_batches <= 0:
                    raise ValueError("Please provide a valid total_validation_batches parameter")

                for i, (batch_x, batch_y) in enumerate(train_data):
                    total_loss = 0
                    total_corrects = 0

                    total_loss, total_corrects = self.__model_loop(batch_x=batch_x, batch_y=batch_y, is_training=True)

                    self.__print_progress(i, total_training_batches, epoch, epochs, total_loss, total_corrects)

                if validation_data:
                    if not isinstance(total_training_batches, int) or total_training_batches <= 0:
                        raise ValueError("Please provide a valid total_training_batches parameter")

                    print("")

                    for i, (batch_x, batch_y) in enumerate(train_data):

                        total_validation_loss = 0
                        total_validation_corrects = 0

                        (total_validation_loss, total_validation_corrects) = self.__model_loop(
                            batch_x=batch_x, batch_y=batch_y, is_training=False
                        )

                        self.__print_progress(
                            i,
                            total_validation_batches,
                            epoch,
                            epochs,
                            total_validation_loss,
                            total_validation_corrects,
                            validation=True,
                        )

                print("")

    def predict(self, predict_data):
        pass

    def predict_classes(self, predict_data):
        pass

    def evaluate(self, evaluate_data, batch_size=None, total_steps=None):
        if isinstance(evaluate_data, tuple):
            batches, labels = evaluate_data

            if not (isinstance(batches, np.ndarray) or isinstance(labels, np.ndarray)):
                raise ValueError("Please provide a valid data to train")

            batches, labels = torch.from_numpy(batches), torch.from_numpy(labels)

            for i in range(0, len(batches), batch_size):
                batch_x = batches[i : i + batch_size]
                batch_y = batches[i : i + batch_size]

                (total_loss, total_corrects) = self.__model_loop(batch_x=batch_x, batch_y=batch_y, is_training=False)

                total_batches = int(len(batches) / batch_size)
                current_batch = int(i / batch_size)

                self.__print_progress(current_batch, total_batches, 0, 1, total_loss, total_corrects, validation=True)
        else:
            for i, (batch_x, batch_y) in enumerate(evaluate_data):
                (total_loss, total_corrects) = self.__model_loop(batch_x=batch_x, batch_y=batch_y, is_training=False)

                self.__print_progress(i // len(batch_x), total_steps, 0, 1, total_loss, total_corrects, validation=True)

    def summary(self, input_shape):
        """Prints a summary of the model with all the layers, number of
        Trainable and Non-Trainable parameters.

        Raises:
            ValueError: If there is no model
        """
        if not self.__model:
            raise ValueError("There is no compiled model to generate a summary")

        summary(self.__model, input_shape)

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
