# Peyton Miller, Homework 1

#Question 1:

#Asked ChatGPT the following:
# "Develop an algorithm in python without numpy that numerically determines the SVD of a matrix, 𝑨. 
# The algorithm should compute the SVD by 
# (i) transforming 𝑨 into a bidiagonal matrix 𝑩 using Householder reflections, 
# (ii) determine the QR-factorization of 𝑩 using Givens rotations to find the SVD of 𝑩, 
# (iii) determine the SVD of 𝑨 using the SVD of 𝑩. Provide specific comments on each step above. 
# You do not need to provide comments on the details of how the Givens rotations or Householder reflections are applied. 
# Apply your algorithm to the matrix below, and provide each matrix component of the decomposition 𝑨 = 𝑼𝚺𝑽⊤. 
# 𝐴 = [ 1 4 9 
#       3 2 3 
#       6 4 5 ]"

import math
import copy


def matmul(A, B):
    m, n, p = len(A), len(B), len(B[0])
    C = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    return [list(row) for row in zip(*A)]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def householder_vector(x):
    v = x[:]
    alpha = norm(v)
    if alpha == 0:
        return v
    if v[0] >= 0:
        alpha = -alpha
    v[0] -= alpha
    beta = norm(v)
    return [vi / beta for vi in v]

def apply_householder_left(A, v, k):
    for i in range(k, len(A)):
        s = sum(v[j-k] * A[j][i] for j in range(k, len(A)))
        for j in range(k, len(A)):
            A[j][i] -= 2 * v[j-k] * s

def apply_householder_right(A, v, k):
    for i in range(len(A)):
        s = sum(v[j-k] * A[i][j] for j in range(k, len(A[0])))
        for j in range(k, len(A[0])):
            A[i][j] -= 2 * v[j-k] * s


def bidiagonalize(A):
    A = copy.deepcopy(A)
    m, n = len(A), len(A[0])
    U = identity(m)
    V = identity(n)

    for k in range(min(m, n)):
       
        x = [A[i][k] for i in range(k, m)]
        v = householder_vector(x)
        apply_householder_left(A, v, k)
        apply_householder_left(U, v, k)

        if k < n - 1:
           
            x = [A[k][j] for j in range(k+1, n)]
            v = householder_vector(x)
            apply_householder_right(A, v, k+1)
            apply_householder_right(V, v, k+1)

    return U, A, V


def givens(a, b):
    r = math.hypot(a, b)
    return a/r, b/r


def svd_bidiagonal(B, iterations=50):
    n = len(B)
    U = identity(n)
    V = identity(n)
    B = copy.deepcopy(B)

    for _ in range(iterations):
        
        for i in range(n-1):
            c, s = givens(B[i][i], B[i+1][i])
            for j in range(n):
                t1 = c*B[i][j] + s*B[i+1][j]
                t2 = -s*B[i][j] + c*B[i+1][j]
                B[i][j], B[i+1][j] = t1, t2

                t1 = c*U[i][j] + s*U[i+1][j]
                t2 = -s*U[i][j] + c*U[i+1][j]
                U[i][j], U[i+1][j] = t1, t2

        for i in range(n-1):
            c, s = givens(B[i][i], B[i][i+1])
            for j in range(n):
                t1 = c*B[j][i] + s*B[j][i+1]
                t2 = -s*B[j][i] + c*B[j][i+1]
                B[j][i], B[j][i+1] = t1, t2

                t1 = c*V[j][i] + s*V[j][i+1]
                t2 = -s*V[j][i] + c*V[j][i+1]
                V[j][i], V[j][i+1] = t1, t2

    Sigma = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Sigma[i][i] = abs(B[i][i])

    return U, Sigma, V






