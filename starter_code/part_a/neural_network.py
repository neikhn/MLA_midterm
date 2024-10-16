# If you encounter pathing error, try the below 3 line for Python to locate the module
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
def load_data(base_path="data"):
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
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # Apply the first linear transformation, followed by sigmoid activation.
        latent = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(latent))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizer.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Loss function: Mean squared error (MSE).
    criterion = nn.MSELoss()

    for epoch in range(num_epoch):
        train_loss = 0.0

        # Iterate over all users
        for user_id in range(train_data.shape[0]):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)  # Shape (1, feature_dim)
            target = inputs.clone()  # Clone to keep the same shape

            # Reset gradients to zero
            optimizer.zero_grad()

            # Forward pass through the model
            output = model(inputs)

            # Create mask for NaN values in the original train_data
            nan_mask = torch.isnan(train_data[user_id])  # Get the mask directly from the PyTorch tensor

            # Ensure that the shapes match before assignment
            if output.shape[1] == target.shape[1]:
                target[0][nan_mask] = output[0][nan_mask]
            else:
                raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")

            # Compute the reconstruction loss
            loss = criterion(output, target)

            # Add L2 regularization term
            weight_norm = model.get_weight_norm()
            loss += lamb * weight_norm

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

        # After each epoch, evaluate on validation set
        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(f"Epoch: {epoch+1}/{num_epoch} \tTraining Loss: {train_loss:.6f}\tValidation Accuracy: {valid_acc:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
# Hyperparameter tuning.
    latent_dims = [10, 50, 100, 200, 500]  # Different values for k.
    lambdas = [0.001, 0.01, 0.1, 1]        # Different values for λ.
    best_valid_acc = 0
    best_model = None

    for k in latent_dims:
        for lamb in lambdas:
            print(f"Training model with k={k}, λ={lamb}...")
            
            # Initialize the model with the current k.
            model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=k)

            # Set hyperparameters.
            lr = 0.01     # You can tune this as well.
            num_epoch = 50  # You can adjust this for your needs.

            # Train the model.
            train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

            # Evaluate on validation set.
            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            
            # Keep track of the best model.
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = model

    print(f"Best Validation Accuracy: {best_valid_acc}")
    # After selecting the best model, you can evaluate it on the test set.
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Test Accuracy with best model: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
