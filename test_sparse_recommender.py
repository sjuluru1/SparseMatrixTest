import pytest
from sparse_recommender import SparseMatrix


def test_set_and_get():
    matrix = SparseMatrix(1, 2)
    matrix.set(1, 2, 3)
    try:
        assert matrix.get(1, 2) == 3
        assert matrix.get(1, 3) == 0  # Test for missing value (should return 0)
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_recommend():
    matrix = SparseMatrix(3, 4)
    matrix.set(0, 0, 1)
    matrix.set(0, 3, 3)
    matrix.set(1, 1, 2)
    matrix.set(1, 3, 1)
    matrix.set(2, 0, 6)
    vector = [1, 2, 3, 4]
    try:
        recommendations = matrix.recommend(vector)
        print(recommendations, 'recommendations')

        # The expected result of matrix-vector multiplication is {2: 12, 1: 2}
        assert (recommendations == [13, 8, 6, 0]).all()
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_add_movie():
    matrix1 = SparseMatrix(3, 3)
    matrix1.set(1, 2, 3)
    matrix1.set(2, 1, 2)

    matrix2 = SparseMatrix(2, 2)
    matrix2.set(1, 0, 1)
    matrix2.set(0, 1, 4)

    try:
        matrix1.add_movie(matrix2)

        assert matrix1.get(0, 1) == 4
        assert matrix1.get(1, 0) == 1  # Value should be updated after adding another matrix
        assert matrix1.get(2, 1) == 2
        assert matrix1.get(1, 2) == 3
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_to_dense():
    matrix = SparseMatrix(3, 3)
    matrix.set(1, 2, 3)
    matrix.set(2, 1, 2)

    try:
        dense_matrix = matrix.to_dense()
        expected_matrix = [[0, 0, 0], [0, 0, 3], [0, 2, 0]]
        for i in range(len(expected_matrix)):
            assert expected_matrix[i] == dense_matrix[i]
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_transpose():
    matrix = SparseMatrix(3, 3)
    matrix.set(0, 1, 3)
    matrix.set(1, 0, 2)
    matrix.set(2, 1, 5)

    try:
        transposed_matrix = matrix.transpose()

        # Check the transposed values
        assert transposed_matrix.get(0, 1) == 2  # Transposed from (1, 0)
        assert transposed_matrix.get(1, 0) == 3  # Transposed from (0, 1)
        assert transposed_matrix.get(1, 2) == 5  # Transposed from (2, 1)

        # Check the dimensions (rows and columns should be swapped)
        assert len(transposed_matrix.matrix) == len(matrix.matrix)  # Number of non-zero elements should be the same
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_get_nonexistent_element():
    matrix = SparseMatrix(0, 0)

    try:
        # Attempt to retrieve a non-existent element, should return 0
        assert matrix.get(0, 0) == 0
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")
    # Attempt to retrieve a non-existent element, should return 0
    # assert matrix.get(0, 0) == 0


def test_set_with_negative_value():
    matrix = SparseMatrix(2, 2)

    try:
        # Setting a negative value should still work
        matrix.set(1, 1, -5)

        # Check that the negative value can be retrieved correctly
        assert matrix.get(1, 1) == -5
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

    # Setting a negative value should still work
    # matrix.set(1, 1, -5)

    # Check that the negative value can be retrieved correctly
    # assert matrix.get(1, 1) == -5


def test_matrix_addition_empty():
    matrix1 = SparseMatrix(0, 0)
    matrix2 = SparseMatrix(0, 0)

    try:
        result_matrix = matrix1.add_movie(matrix2)
        print(result_matrix, 'addition of matrices')

        # Check that the result_matrix is an instance of SparseMatrix
        assert isinstance(result_matrix, SparseMatrix)

        # Check the result of matrix addition
        assert result_matrix.get(0, 0) == 0
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")


def test_matrix_multiplication_zero_vector():
    matrix = SparseMatrix(2, 2)
    matrix.set(0, 0, 2)
    matrix.set(0, 1, 3)
    matrix.set(1, 0, 4)
    matrix.set(1, 1, 5)
    zero_vector = [0, 0]
    try:
        result = matrix.recommend(zero_vector)
        print(result, 'result of recommendation matrix')
        # Check the result of matrix-vector multiplication
        assert (result == [0, 0]).all()
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")
    #
    # result = matrix.recommend(zero_vector)
    #
    # # Check the result of matrix-vector multiplication
    # assert result == {0: 0, 1: 0}


def test_get_negative_index():
    matrix = SparseMatrix(2, 3)
    matrix.set(1, 2, 3)

    try:
        # Accessing a negative index
        matrix.get(-1, -2)
        print(matrix.get(-1, -2), 'actual')
    except ValueError as e:
        actual_message = str(e)
        print(f"Actual exception message: {actual_message}", 'actual')
        expected_message = "Negative indices are not allowed"
        assert expected_message in actual_message, f"Unexpected exception message: {actual_message}"


if __name__ == '__main__':
    pytest.main()
