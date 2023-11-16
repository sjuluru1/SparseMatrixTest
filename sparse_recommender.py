import numpy as np

class SparseMatrix:
    def __init__(self, rows, columns):
        self.matrix = {}  # Use a dictionary to store non-zero elements
        self.columns = columns
        self.rows = rows

    def set(self, row, col, value):
        try:
            if not isinstance(row, int) or not isinstance(col, int):
                raise ValueError("Row and column must be integers")
            if value != 0:
                self.matrix[(row, col)] = value
            elif (row, col) in self.matrix:
                del self.matrix[(row, col)]  # Remove the entry if value is 0
        except ValueError as ve:
            print(f"ValueError occurred: {ve}")
        except Exception as e:
            print(f"Exception occurred: {e}")

    def get(self, row, col):
        try:
            if not isinstance(row, int) or not isinstance(col, int):
                raise ValueError("Row and column must be integers")
            return self.matrix.get((row, col), 0)  # Return 0 if the entry doesn't exist
        except ValueError as ve:
            print(f"ValueError occurred: {ve}")
            return 0
        except Exception as e:
            print(f"Exception occurred: {e}")
            return 0

    def recommend(self, vector):
        try:
            if len(vector) != self.columns:
                raise ValueError("Vector length must match the number of columns in the matrix")

            result = np.zeros(len(vector))  # Initialize the result vector with zeros

            for (i, j), value in self.matrix.items():
                if i >= len(result) or j >= len(vector):
                    raise ValueError("Matrix indices out of bounds")

                result[i] += value * vector[j]

            return result
        except Exception as e:
            raise e

    def add_movie(self, matrix):
        try:
            if not isinstance(matrix, SparseMatrix):
                raise ValueError("Input matrix must be an instance of SparseMatrix")
            for (row, col), value in matrix.matrix.items():
                existing_value = self.get(row, col)
                self.set(row, col, existing_value + value)
            return self
        except ValueError as ve:
            print(f"ValueError occurred: {ve}")
            return self
        except Exception as e:
            print(f"Exception occurred: {e}")
            return self

    def to_dense(self):
        try:
            max_row, max_col = 0, 0
            for row, col in self.matrix.keys():
                max_row = max(max_row, row)
                max_col = max(max_col, col)

            dense_matrix = np.zeros((max_row + 1, max_col + 1), dtype=int)

            for (row, col), value in self.matrix.items():
                dense_matrix[row][col] = value

            return dense_matrix.tolist()
        except Exception as e:
            print(f"Exception occurred: {e}")
            return []

    def transpose(self):
        try:
            transposed_matrix = SparseMatrix(self.rows, self.columns)
            for (row, col), value in self.matrix.items():
                transposed_matrix.set(col, row, value)
            return transposed_matrix
        except Exception as e:
            print(f"Exception occurred: {e}")
            return SparseMatrix(self.rows, self.columns)

