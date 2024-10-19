from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc

def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    """ Fill in missing values using k-nearest neighbors, weighting neighbors based on their similarity."""
    
    num_students, num_questions = matrix.shape
    question_similarity = np.zeros((num_questions, num_questions))

    # Calculate similarity between questions using cosine similarity
    for i in range(num_questions):
        for j in range(i + 1, num_questions):
            valid_entries = (~np.isnan(matrix[:, i])) & (~np.isnan(matrix[:, j]))
            if np.any(valid_entries):
                dot_product = np.dot(matrix[valid_entries, i], matrix[valid_entries, j])
                norm_i = np.linalg.norm(matrix[valid_entries, i])
                norm_j = np.linalg.norm(matrix[valid_entries, j])
                if norm_i > 0 and norm_j > 0:
                    question_similarity[i, j] = dot_product / (norm_i * norm_j)
                    question_similarity[j, i] = question_similarity[i, j]

    matrix_imputed = matrix.copy()

    # Impute missing values using weighted k-nearest neighbors
    for question_id in range(num_questions):
        missing_students = np.isnan(matrix[:, question_id])

        for student_id in np.where(missing_students)[0]:
            answered_questions = ~np.isnan(matrix[student_id, :])
            similarity_scores = question_similarity[question_id, answered_questions]
            distances = 1 - similarity_scores  # Using distance as 1 - similarity

            if len(similarity_scores) > 0:
                top_k_indices = np.argsort(similarity_scores)[-k:]
                top_k_similarities = similarity_scores[top_k_indices]
                top_k_answers = matrix[student_id, answered_questions][top_k_indices]

                # Apply weighted average with weight decay
                weights = top_k_similarities / (1 + distances[top_k_indices])
                if np.sum(weights) > 0:
                    matrix_imputed[student_id, question_id] = np.dot(weights, top_k_answers) / np.sum(weights)
                else:
                    matrix_imputed[student_id, question_id] = np.mean(top_k_answers)

    acc = sparse_matrix_evaluate(valid_data, matrix_imputed)
    print("Validation Accuracy with Weighting: {}".format(acc)) 
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def knn_impute_by_item_weighted(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on weighted question similarity."""
    num_students, num_questions = matrix.shape
    question_similarity = np.zeros((num_questions, num_questions))

    # Calculate similarity between questions using cosine similarity
    for i in range(num_questions):
        for j in range(i + 1, num_questions):
            valid_entries = (~np.isnan(matrix[:, i])) & (~np.isnan(matrix[:, j]))
            if np.any(valid_entries):
                dot_product = np.dot(matrix[valid_entries, i], matrix[valid_entries, j])
                norm_i = np.linalg.norm(matrix[valid_entries, i])
                norm_j = np.linalg.norm(matrix[valid_entries, j])
                if norm_i > 0 and norm_j > 0:
                    question_similarity[i, j] = dot_product / (norm_i * norm_j)
                    question_similarity[j, i] = question_similarity[i, j]

    matrix_imputed = matrix.copy()

    # Impute missing values using weighted k-nearest neighbors
    for question_id in range(num_questions):
        missing_students = np.isnan(matrix[:, question_id])

        for student_id in np.where(missing_students)[0]:
            answered_questions = ~np.isnan(matrix[student_id, :])
            similarity_scores = question_similarity[question_id, answered_questions]

            if len(similarity_scores) > 0:
                top_k_indices = np.argsort(similarity_scores)[-k:]  # Get indices of top k similar questions
                top_k_similarities = similarity_scores[top_k_indices]
                top_k_answers = matrix[student_id, answered_questions][top_k_indices]

                # Apply weighted average based on similarity scores
                if np.sum(top_k_similarities) > 0:
                    weights = top_k_similarities / np.sum(top_k_similarities)
                    matrix_imputed[student_id, question_id] = np.dot(weights, top_k_answers)
                else:
                    # If all similarities are 0, fall back to unweighted average
                    matrix_imputed[student_id, question_id] = np.mean(top_k_answers)

    acc = sparse_matrix_evaluate(valid_data, matrix_imputed)
    print("Validation Accuracy with Weighting (Item-based, k={}): {}".format(k, acc))
    return acc

def main():
    # Verify the correct data path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train_sparse.npz")
    if not os.path.exists(data_path):
        raise Exception(f"The specified path {data_path} does not exist.")

    sparse_matrix = load_train_sparse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")).toarray()
    val_data = load_valid_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    test_data = load_public_test_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

    print("Sparse matrix shape: ", sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]

    # Store accuracies for plotting
    user_accs = []
    item_accs = []
    item_accs_with_weights = []

    # Best k's and accuracies
    best_k_user, best_acc_user = None, 0
    best_k_item, best_acc_item = None, 0
    best_k_item_weighted, best_acc_item_weighted = None, 0

    # User-based KNN
    print("\nUser-based KNN Imputation (validation stage):")
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accs.append(acc)
        if acc > best_acc_user:
            best_acc_user = acc
            best_k_user = k

    # Item-based KNN
    print("\nItem-based KNN Imputation (validation stage):")
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accs.append(acc)
        if acc > best_acc_item:
            best_acc_item = acc
            best_k_item = k

    # Item-based KNN with weighting
    print("\nItem-based KNN Imputation with Weighting (validation stage):")
    for k in k_values:
        acc = knn_impute_by_item_weighted(sparse_matrix, val_data, k)
        item_accs_with_weights.append(acc)
        if acc > best_acc_item_weighted:
            best_acc_item_weighted = acc
            best_k_item_weighted = k

    # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, user_accs, label="User-based KNN", marker='o')
    plt.plot(k_values, item_accs, label="Item-based KNN", marker='s')
    plt.plot(k_values, item_accs_with_weights, label="Item-based KNN with Weighting", marker='x')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs k for KNN Imputation Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on the test data
    print("\nEvaluating on test data with best k values:")
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print(f"Test Accuracy (User-based, k={best_k_user}): {user_test_acc}")

    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print(f"Test Accuracy (Item-based, k={best_k_item}): {item_test_acc}")

    item_test_acc_weighted = knn_impute_by_item_weighted(sparse_matrix, test_data, best_k_item_weighted)
    print(f"Test Accuracy (Item-based with Weighting, k={best_k_item_weighted}): {item_test_acc_weighted}")


if __name__ == "__main__":
    main()
