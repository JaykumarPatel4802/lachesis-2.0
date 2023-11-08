import numpy as np
import argparse

def matrix_multiply(matrix_size):
    matrix_A = np.random.rand(matrix_size, matrix_size)
    matrix_B = np.random.rand(matrix_size, matrix_size)

    # Perform matrix multiplication
    result_matrix = np.matmul(matrix_A, matrix_B)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Multiplication")
    parser.add_argument("size", type=int, help="Size of the square matrices")
    args = parser.parse_args()

    matrix_multiply(args.size)
