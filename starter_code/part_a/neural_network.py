# If you have pathing error, add the 3 below lines 
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

class AutoEncoder(nn.Module):
    def __init__(self, num_question, latent_dim=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param latent_dim: int
        """
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Linear(num_question, latent_dim)
        self.decoder = nn.Linear(latent_dim, num_question)

    def get_weight_norm(self):
        """ Return the sum of L2 norms of the weights for encoder and decoder.

        :return: float
        """
        encoder_norm = torch.norm(self.encoder.weight, 2) ** 2
        decoder_norm = torch.norm(self.decoder.weight, 2) ** 2
        return encoder_norm + decoder_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        # Encode the input with sigmoid activation.
        encoded = torch.sigmoid(self.encoder(inputs))
        # Decode the latent representation back to the output space using sigmoid.
        reconstructed = torch.sigmoid(self.decoder(encoded))
        return reconstructed


def train(model, learning_rate, reg_param, train_data, zero_train_data, valid_data, epochs):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param learning_rate: float
    :param reg_param: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param epochs: int
    :return: None
    """
    model.train()  # Set the model to training mode
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loss_list = []
    valid_acc_list = []

    for epoch in range(epochs):
        cumulative_loss = 0.0
        for user_id in range(train_data.shape[0]):
            input_vector = Variable(zero_train_data[user_id]).unsqueeze(0)
            target_vector = input_vector.clone()

            optimizer.zero_grad()
            predicted_output = model(input_vector)

            missing_mask = torch.isnan(train_data[user_id])
            if predicted_output.shape[1] == target_vector.shape[1]:
                target_vector[0][missing_mask] = predicted_output[0][missing_mask]
            else:
                raise ValueError(f"Output shape {predicted_output.shape} does not match target shape {target_vector.shape}")
            
            # Calculate loss
            loss = criterion(predicted_output, target_vector)
            regularization_loss = model.get_weight_norm()
            total_loss = loss + reg_param * regularization_loss
            
            total_loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            cumulative_loss += total_loss.item()

        # Track validation accuracy
        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_loss_list.append(cumulative_loss)
        valid_acc_list.append(valid_acc)
        print(f"Epoch: {epoch + 1}/{epochs} \tTraining Loss: {cumulative_loss:.6f}\tValidation Accuracy: {valid_acc:.4f}")

    # Plot loss and accuracy graphs
    plot_graph(train_loss_list, valid_acc_list)


def plot_graph(train_loss, valid_acc):
    """Plot the training loss and validation accuracy"""
    epochs = len(train_loss)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), valid_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')

    plt.tight_layout()
    plt.show()


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0

    for i, user_id in enumerate(valid_data["user_id"]):
        input_vector = Variable(train_data[user_id]).unsqueeze(0)
        predicted_output = model(input_vector)

        prediction = predicted_output[0][valid_data["question_id"][i]].item() >= 0.5
        if prediction == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # latent_dimensions = [1000]
    # regularization_params = [0.001]
    latent_dimensions = [1, 10, 100, 500, 1000]
    regularization_params = [0.001, 0.01, 0.1, 1]
    best_valid_accuracy = 0
    optimal_model = None
    
    for latent_dim in latent_dimensions:
        for reg_param in regularization_params:
            print(f"Training model with latent dimension={latent_dim}, λ={reg_param}...")
            model = AutoEncoder(num_question=zero_train_matrix.shape[1], latent_dim=latent_dim)
            learning_rate = 0.02
            epochs = 50
            train(model, learning_rate, reg_param, train_matrix, zero_train_matrix, valid_data, epochs)
            validation_accuracy = evaluate(model, zero_train_matrix, valid_data)

            if validation_accuracy > best_valid_accuracy:
                best_valid_accuracy = validation_accuracy
                optimal_model = model

    print(f"Best Validation Accuracy: {best_valid_accuracy}")
    test_accuracy = evaluate(optimal_model, zero_train_matrix, test_data)
    print(f"Test Accuracy with the optimal model: {test_accuracy}")


if __name__ == "__main__":
    main()