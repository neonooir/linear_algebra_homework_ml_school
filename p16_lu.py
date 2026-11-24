import numpy as np
import time
rng = np.random.default_rng(0)


def lu(A):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    for c in range(n - 1):
        for i in range(c + 1, n):
            L[i, c] = U[i, c] / U[c, c]
            U[i, c:] = U[i, c:] - L[i, c] * U[c, c:]
    return L, U


def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


print("Part 1: the example from the notes")
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
L, U = lu(A)
print("L =\n", L)
print("U =\n", U)
b = np.array([5, 13, 31])
print("solution of A x = (5, 13, 31):", solve_lu(L, U, b))

ok = True
for t in range(10):
    A = rng.normal(size=(6, 6))
    L, U = lu(A)
    if not np.allclose(L @ U, A):
        ok = False
    if not np.allclose(L, np.tril(L)) or not np.allclose(U, np.triu(U)):
        ok = False
print("L @ U == A for random 6x6 matrices, L lower, U upper:", ok)

print("\nPart 2: solve_lu vs np.linalg.solve")
ok = True
for t in range(10):
    A = rng.normal(size=(6, 6))
    b = rng.normal(size=6)
    L, U = lu(A)
    if not np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b)):
        ok = False
print("agrees with np.linalg.solve:", ok)

A = np.array([[1e-17, 1.0], [1.0, 1.0]])
b = np.array([1.0, 2.0])
L, U = lu(A)
print("tiny pivot, no swap:   x =", solve_lu(L, U, b))
A_swap = A[[1, 0]]
b_swap = b[[1, 0]]
L, U = lu(A_swap)
print("rows swapped first:    x =", solve_lu(L, U, b_swap))

print("\nPart 3: Sherman-Morrison updates")
X = rng.normal(size=(200, 50))
new_rows = rng.normal(size=(100, 50))

start = time.perf_counter()
Ainv = np.linalg.inv(X.T @ X)
for k in range(100):
    u = new_rows[k]
    Au = Ainv @ u
    Ainv = Ainv - np.outer(Au, Au) / (1 + u @ Au)
time_sm = time.perf_counter() - start

start = time.perf_counter()
X_all = X
for k in range(100):
    X_all = np.vstack([X_all, new_rows[k]])
    Ainv_scratch = np.linalg.inv(X_all.T @ X_all)
time_scratch = time.perf_counter() - start

print("updated inverse == inverse from scratch:", np.allclose(Ainv, Ainv_scratch))
print(f"time: Sherman-Morrison {time_sm * 1000:.2f} ms, from scratch each time {time_scratch * 1000:.2f} ms")

print("""
Answers:
1. lu reproduces the example: L = [[1,0,0],[2,1,0],[4,3,1]], U = [[2,1,1],[0,1,1],[0,0,2]], x = (1, 2, 1).
2. For A = [[1e-17, 1], [1, 1]] the solver returns x = (0, 1) instead of about (1, 1). The pivot 1e-17 gives
   the multiplier 1e17, so U_22 = 1 - 1e17, which the computer rounds to -1e17: the entry 1 of A is lost.
   Back substitution then gives x_2 = 1 and x_1 = (1 - 1) / 1e-17 = 0. After swapping the rows, the pivot is 1,
   the multiplier is 1e-17, nothing important is rounded away, and the solver returns (1, 1).
3. Adding a row u^T to X changes X^T X to X^T X + u u^T, a rank-one update, so Sherman-Morrison gives the new
   inverse with about n^2 operations instead of n^3 for a new inversion. The results agree, and the updates are
   much faster than recomputing (X^T X)^{-1} after every new row.
""")
