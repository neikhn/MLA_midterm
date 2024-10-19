from utils import *
import numpy as np
from part_a.item_response import *

"""
IRT
"""


def bootstrap_data(data):
    """Generate a bootstrapped version of the training data"""
    n_samples = len(data["user_id"])
    indices = np.random.choice(n_samples, n_samples, replace=True)

    bootstrapped = {key: np.array([data[key][i] for i in indices]) for key in data}
    return bootstrapped


def bagging_ensemble(data, val_data, test_data, lr, iterations, n_models=3):
    """
    Train an ensemble of IRT models using bagging and evaluate their performance.

    :param data: Training data
    :param val_data: Validation data
    :param test_data: Test data
    :param lr: Learning rate
    :param iterations: Number of iterations
    :param n_models: Number of models for bagging ensemble
    :return: Test accuracy of the ensemble
    """

    models = []
    for i in range(n_models):
        print(f"Model: {i}")
        bootstrapped = bootstrap_data(data)
        theta, beta, _ = irt(bootstrapped, val_data, lr, iterations)
        models.append((theta, beta))

    val_acc, acc_models = evaluate_ensemble(val_data, models)
    test_acc, test_acc_models = evaluate_ensemble(test_data, models)

    for i in range(n_models):
        print(f"Model {i+1}: Val Acc: {acc_models[i]}, Test Acc: {test_acc_models[i]} ")

    print(f"Ensemble Validation Accuracy: {val_acc}")
    print(f"Ensemble Test Accuracy: {test_acc}")

    return test_acc


def evaluate_ensemble(data, models):
    """
    Evaluate the ensemble by averaging the predictions from multiple models.
    :param data: Validation or test data
    :param models: List of (theta, beta) tuples for each model
    :return: Accuracy of the ensemble
    """

    n_models = len(models)
    ensemble_pred = np.zeros(len(data["is_correct"]))
    acc_models = []

    for theta, beta in models:
        pred = []
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            x = theta[u] - beta[q]
            p_a = sigmoid(x)
            pred.append(p_a >= 0.5)
        acc_model = np.mean(data["is_correct"] == np.array(pred))
        acc_models.append(acc_model)
        ensemble_pred += np.array(pred)

    final_pred = (ensemble_pred / n_models) >= 0.5
    acc = np.sum(data["is_correct"] == final_pred) / len(data["is_correct"])

    return acc, acc_models


def main():
    train_data = load_train_csv("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    lr = 0.0006
    iterations = 250
    n_models = 3

    bagging_ensemble(train_data, val_data, test_data, lr, iterations, n_models)


if __name__ == "__main__":
    main()
