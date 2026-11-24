import numpy as np
rng = np.random.default_rng(0)


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        best = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[best, c]) < tol:
            continue
        R[[r, best]] = R[[best, r]]
        R[r] = R[r] / R[r, c]
        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    R[np.abs(R) < tol] = 0.0
    return R, pivots


def rank(A):
    R, pivots = rref(A)
    return len(pivots)


def null_space_basis(A):
    R, pivots = rref(A)
    n = R.shape[1]
    free = [j for j in range(n) if j not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n)
        v[f] = 1.0
        for i, p in enumerate(pivots):
            v[p] = -R[i, f]
        basis.append(v)
    return basis


def col_space_basis(A):
    A = np.array(A, dtype=float)
    R, pivots = rref(A)
    return [A[:, p] for p in pivots]


print("Test 1")
R, pivots = rref([[1, 2, -1], [3, -1, 4]])
print(R)
print("pivots:", pivots)

print("\nTest 2: matrix F")
F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
R, pivots = rref(F)
print(R)
print("pivots:", pivots)

print("\nTest 3: 20 random 5x7 matrices")
ok = True
for t in range(20):
    A = rng.normal(size=(5, 7))
    N = null_space_basis(A)
    for v in N:
        if not np.allclose(A @ v, 0):
            ok = False
    if rank(A) + len(N) != 7:
        ok = False
    if rank(A) != np.linalg.matrix_rank(A):
        ok = False
print("A v = 0, rank-nullity and matrix_rank all agree:", ok)

print("\nTest 4: 6x8 matrix of rank 4")
A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
N = null_space_basis(A)
print("rank(A) =", rank(A))
print("dim null(A) =", len(N))
print("A v = 0 for all v in the basis:", all(np.allclose(A @ v, 0) for v in N))
print("number of column space basis vectors:", len(col_space_basis(A)))
