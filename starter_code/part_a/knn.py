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
    num_students, num_questions = matrix.shape
    
    # Step 1: Calculate the similarity between questions (columns in the matrix)
    # Using cosine similarity for similarity between questions
    question_similarity = np.zeros((num_questions, num_questions))
    
    for i in range(num_questions):
        for j in range(i + 1, num_questions):
            valid_entries = (~np.isnan(matrix[:, i])) & (~np.isnan(matrix[:, j]))  # Mask to filter valid entries
            if np.any(valid_entries):  # Check if there are any valid comparisons
                dot_product = np.dot(matrix[valid_entries, i], matrix[valid_entries, j])
                norm_i = np.linalg.norm(matrix[valid_entries, i])
                norm_j = np.linalg.norm(matrix[valid_entries, j])
                if norm_i > 0 and norm_j > 0:  # Avoid division by zero
                    question_similarity[i, j] = dot_product / (norm_i * norm_j)
                    question_similarity[j, i] = question_similarity[i, j]  # Symmetry

    # Step 2: Impute missing values based on k-nearest neighbors (similar questions)
    matrix_imputed = matrix.copy()

    for question_id in range(num_questions):
        missing_students = np.isnan(matrix[:, question_id])  # Find missing values for the current question

        for student_id in np.where(missing_students)[0]:
            # Find the k most similar questions that the student has answered
            answered_questions = ~np.isnan(matrix[student_id, :])
            similarity_scores = question_similarity[question_id, answered_questions]
            
            if len(similarity_scores) > 0:
                # Get the indices of the k most similar questions
                top_k_indices = np.argsort(similarity_scores)[-k:]  # Indices of k most similar questions
                top_k_similarities = similarity_scores[top_k_indices]
                top_k_answers = matrix[student_id, answered_questions][top_k_indices]

                # Weighted average to impute the missing value
                if np.sum(top_k_similarities) > 0:
                    matrix_imputed[student_id, question_id] = np.dot(top_k_similarities, top_k_answers) / np.sum(top_k_similarities)
                else:
                    # If all similarities are zero, use a simple average of the top k answers
                    matrix_imputed[student_id, question_id] = np.mean(top_k_answers)
    
    # Step 3: Evaluate the accuracy of the imputed matrix
    acc = sparse_matrix_evaluate(valid_data, matrix_imputed)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    # Verify the correct data path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train_sparse.npz")
    if not os.path.exists(data_path):
        raise Exception(f"The specified path {data_path} does not exist.")

    sparse_matrix = load_train_sparse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")).toarray()
    val_data = load_valid_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    test_data = load_public_test_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Test different k values for both user-based and item-based filtering.

    k_values = [1, 6, 11, 16, 21, 26]
    
    # Store accuracies for plotting
    user_accs = []
    item_accs = []

    # User-based KNN
    best_k_user = None
    best_acc_user = 0
    print("\nUser-based KNN Imputation (validation stage):")
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        if acc is not None:  # Ensure the function returns a valid accuracy
            user_accs.append(acc)
        else:
            print(f"User-based KNN returned None for k={k}.")
        if acc > best_acc_user:
            best_acc_user = acc
            best_k_user = k

    print(f"\nBest User-based KNN validation accuracy: {best_acc_user} (k = {best_k_user})")
    print("User-based accuracies:", user_accs)
    
    # Item-based KNN
    best_k_item = None
    best_acc_item = 0
    print("\nItem-based KNN Imputation (validation stage):")
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        if acc is not None:  # Ensure the function returns a valid accuracy
            item_accs.append(acc)
        else:
            print(f"Item-based KNN returned None for k={k}.")
        if acc > best_acc_item:
            best_acc_item = acc
            best_k_item = k

    print(f"\nBest Item-based KNN validation accuracy: {best_acc_item} (k = {best_k_item})")
    print("Item-based accuracies:", item_accs)
    
    # Check that the accuracy lists have the correct lengths before plotting
    if len(user_accs) != len(k_values) or len(item_accs) != len(k_values):
        print("Error: The length of accuracies does not match the length of k_values.")
        print(f"user_accs: {len(user_accs)}, item_accs: {len(item_accs)}, k_values: {len(k_values)}")
        return
    
    # Plotting the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, user_accs, label="User-based KNN", marker='o')
    plt.plot(k_values, item_accs, label="Item-based KNN", marker='s')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs k for User-based and Item-based KNN')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model on the test data using the best k found for both methods.
    print("\nEvaluating on test data with best k:")
    
    # User-based KNN test accuracy
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print(f"Test Accuracy (User-based, k={best_k_user}): {user_test_acc}")

    # Item-based KNN test accuracy
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print(f"Test Accuracy (Item-based, k={best_k_item}): {item_test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()