import numpy as np

weights = np.array([0.1, 0.2, 0])


def neural_network(input, weights):
    pred = input.dot(weights)
    return pred


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

# Input corresponds to every entry
# for the first game of the season.

input = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input, weights)

print(pred)

# Instead of predicting just
# whether the team won or lost,
# now we're also predicting whether
# they are happy/sad AND the percentage
# of the team that is hurt. We are
# making this prediction using only
# the current win/loss record.


def ele_mul(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


weights = [0.3, 0.2, 0.9]


def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred


wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
pred = neural_network(input, weights)

print(pred)


# toes %win #fans
weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]  # hurt?  # win?  # sad?


def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output


def vect_mat_mul(vect, matrix):
    assert len(vect) == len(matrix)
    output = [0, 0, 0]
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])
    return output


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred


# This dataset is the current
# status at the beginning of
# each game for the first 4 games
# in a season.

# toes = current number of toes
# wlrec = current games won (percent)
# nfans = fan count (in millions)

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

# Input corresponds to every entry
# for the first game of the season.

input = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input, weights)

print(pred)
