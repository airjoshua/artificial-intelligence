import numpy as np


def dot_product(vector_1, vector_2):
    # Use a breakpoint in the code line below to debug your script.
    return sum(a * b for a, b in zip(vector_1, vector_2))


def matrix_multiplication(matrix_a, matrix_b):
    assert matrix_a.shape == matrix_b.shape
    m, n = matrix_a.shape
    x = np.zeros(shape=(m, n))
    for i in range(m):
        for j in range(n):
            x[i, j] = dot_product(matrix_a[i, :], matrix_b[:, j])
    pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    a = np.array([[1, 0], [0, 1]])
    b = np.array([[4, 1], [2, 2]])
    matrix_multiplication(a, b)
    dot_product(vector_1=[1, 2, 3], vector_2=[4, 5, 6])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
