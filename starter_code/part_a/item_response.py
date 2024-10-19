import matplotlib
from matplotlib import pyplot as plt

from utils import *

matplotlib.use('TkAgg')


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for i, q in enumerate(data['question_id']):
        u = data['user_id'][i]
        c = data['is_correct'][i]
        theta_u = theta[u]
        beta_q = beta[q]
        # Calculate the probability using the sigmoid function
        p = sigmoid(theta_u - beta_q)
        # Accumulate the log-likelihood
        log_lklihood += c * np.log(p) + (1 - c) * np.log(1 - p)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    for i, q in enumerate(data['question_id']):
        u = data['user_id'][i]
        c = data['is_correct'][i]
        theta_u = theta[u]
        beta_q = beta[q]
        p = sigmoid(theta_u - beta_q)

        # Update the gradients
        theta[u] += lr * (c - p)
        beta[q] -= lr * (c - p)
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Initialize theta and beta.
    theta = np.random.randn(max(data["user_id"]) + 1)
    beta = np.random.randn(max(data["question_id"]) + 1)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("Iteration: {} \t NLLK: {} \t Score: {}".format(i, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot_prob_vs_theta(theta, beta, question_ids):
    """ Plot the probability of correct response as a function of student ability (theta)
        for the given questions.

    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :param question_ids: List of question ids to plot for
    """
    theta_range = np.linspace(-4, 4, 100)  # Range of student abilities (θ)

    plt.figure(figsize=(10, 6))

    for q_id in question_ids:
        # For each question, calculate the probability of correct response as a function of θ
        probabilities = sigmoid(theta_range - beta[q_id])
        plt.plot(theta_range, probabilities, label=f"Question {q_id}")

    plt.title("Probability of Correct Response vs Student Ability (θ)")
    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    # Tune learning rate and number of iterations
    lr = 0.01  # Learning rate
    iterations = 100  # Number of iterations
    # Train the model
    theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)
    # Evaluate on the test data
    test_acc = evaluate(test_data, theta, beta)
    print("Final Test Accuracy: {}".format(test_acc))

    # Select three distinct questions
    question_ids = [1525, 773, 1103]

    # Plot the probability as a function of θ for the selected questions
    plot_prob_vs_theta(theta, beta, question_ids)


if __name__ == "__main__":
    main()
