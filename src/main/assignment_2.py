# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:54:38 2023

Assignment 2

Jacob Romero
course: Numerical Calculus
date: 1/23/2023
"""

'''
1. Using Neville’s method, find the 2nd degree interpolating value for f(3.7) for the following 
set of data
'''

import numpy as np


def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((3, 3))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # the end of the first loop are how many columns you have...
    num_of_points = len(x_points)

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]

            denominator = x_points[i] - x_points[i-j]

            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication)/denominator
            matrix[i][j] = coefficient

    print(matrix[len(x_points)-1][len(x_points)-1], "\n")
    return None


if __name__ == "__main__":
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7

    nevilles_method(x_points, y_points, approximating_value)
    

'''
2. Using Newton’s forward method, print out the polynomial approximations for degrees 1, 2, 
and 3 using the following set of data
'''

import numpy as np

def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((4,4))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            matrix[i][j] = operation

    return matrix


def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    # final result
    return reoccuring_px_result


if __name__ == "__main__":
    # point setup
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference_table(x_points, y_points)

    Array = []
    for i in range(1, len(x_points)):
        Array.append(divided_table[i][i])
    print(Array, "\n")

'''
3. Using the results from 2, approximate f(7.3)?
'''
# find approximation
approximating_x = 7.3
final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
print(final_approximation, "\n")

'''
4. Using the divided difference method, print out the Hermite polynomial approximation matrix
'''

import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # get left cell entry
            left: float = matrix[i][j-1]

            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = (left - diagonal_left)

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i-j+1][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]

    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = 2*len(x_points)
    matrix = np.zeros((num_of_points, num_of_points))

    # populate x values (make sure to fill every TWO rows)
    counter = 0
    for x in range(0, num_of_points, 2):
        matrix[x][0] = x_points[counter]
        matrix[x+1][0] = x_points[counter]
        counter += 1
    
    # prepopulate y values (make sure to fill every TWO rows)
    counter = 0
    for x in range(0, num_of_points, 2):
        matrix[x][1] = y_points[counter]
        matrix[x+1][1] = y_points[counter]
        counter += 1

    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    counter = 0
    for x in range(1, num_of_points, 2):
        matrix[x][2] = slopes[counter]
        counter += 1

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix, "\n")


if __name__ == "__main__":
    hermite_interpolation()
    

'''
5. Using cubic spline interpolation, solve for the following using this set of data:
'''
# a) Find matrix A

import numpy as np

x_points = np.array([2, 5, 8, 10])
y_points = np.array([3, 5, 7, 9])

# Define the matrix A
n = len(x_points)
A = np.zeros((n, n))
A[0, 0] = 1
A[n-1, n-1] = 1

for i in range(1, n-1):
    A[i, i-1] = x_points[i] - x_points[i-1]
    A[i, i] = 2 * (x_points[i+1] - x_points[i-1])
    A[i, i+1] = x_points[i+1] - x_points[i]

print(A)

# b) Vector b

# Define the vector b
b = np.zeros(n)
for i in range(1, n-1):
    b[i] = 3 * (y_points[i+1] - y_points[i]) / (x_points[i+1] - x_points[i]) - \
           3 * (y_points[i] - y_points[i-1]) / (x_points[i] - x_points[i-1])

print(b)

# c) Vector x

# Solve for the vector x
x = np.linalg.solve(A, b)

print(x)