import os, json
import torch
import torch.nn as nn
from gbmhackathon.utils.module_functions import enforce_signature_types

class BaseModule(nn.Module):
    """
    Base class for all our models. It allows us to define methods common to all of them.
    """

    def __init__(self):
        super().__init__()
        self.history = {"epochs": [], "test": []}

    def train_log(
        self, train_batch_losses, val_batch_losses, train_loss, validation_loss
    ):
        """Log batch losses and per batch average loss during training for training and validation batches"""
        self.history["epochs"].append(
            {
                "train_batch_losses": train_batch_losses,
                "val_batch_losses": val_batch_losses,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

    def test_log(self, test_batch_losses, test_loss):
        """Log batch losses and per batch average loss at test time"""
        self.history["test"].append(
            {"test_batch_losses": test_batch_losses, "test_loss": test_loss}
        )

    def save_model(self, directory: str, name: str):
        """
        Saves the model architecture and state using state-of-the-art PyTorch methods.

        Parameters:
            path (str): The path to save the model file.
        """
        # Making sure the directory exist, if not, creates it
        os.makedirs(directory, exist_ok=True)

        # Save state dictionary.
        model_weights_path = f"{directory}/{name}.pt"
        torch.save(self.state_dict(), model_weights_path)
        print(f"Model saved successfully in {directory}")

        # Saving class name to be able to know precisely which class was used
        self.history["class_name"] = str(self.__class__)
        # save history
        history_path = os.path.join(directory, f"{name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f)


class TestClass(BaseModule, nn.Module):
    @enforce_signature_types
    def __init__(self, conv_size=3):
        super().__init__()
        self.conv_layer = nn.Conv1d(1, 1, conv_size)

    def forward(self, x):
        return self.conv_layer(x)
