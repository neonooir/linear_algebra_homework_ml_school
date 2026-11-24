import numpy as np
import time
rng = np.random.default_rng(0)

print("Part 1: reshape")
T = np.arange(24).reshape(2, 3, 4)
flat = T.reshape(-1)
ok = True
for i in range(2):
    for j in range(3):
        for k in range(4):
            if T[i, j, k] != flat[12 * i + 4 * j + k]:
                ok = False
print("T[i, j, k] == T.reshape(-1)[12i + 4j + k] for all indices:", ok)
U1 = T.reshape(2, 12)
U2 = T.transpose(1, 0, 2).reshape(3, 8)
print("T.reshape(2, 12):\n", U1)
print("rank =", np.linalg.matrix_rank(U1))
print("T.transpose(1, 0, 2).reshape(3, 8):\n", U2)
print("rank =", np.linalg.matrix_rank(U2))

print("\nPart 2: the operations of the einsum table")
A = rng.normal(size=(3, 4))
B = rng.normal(size=(4, 5))
S = rng.normal(size=(4, 4))
x = rng.normal(size=4)
y = rng.normal(size=4)
print("matrix product AB:", np.allclose(np.einsum("ij,jk->ik", A, B), A @ B))
print("trace:            ", np.isclose(np.einsum("ii->", S), np.trace(S)))
print("dot product:      ", np.isclose(np.einsum("i,i->", x, y), x @ y))
print("outer product:    ", np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y)))
print("transpose:        ", np.allclose(np.einsum("ij->ji", A), A.T))
print("bilinear form:    ", np.isclose(np.einsum("i,ij,j->", x, S, y), x @ S @ y))

As = rng.normal(size=(100, 4, 4))
Bs = rng.normal(size=(100, 4, 4))

start = time.perf_counter()
traces_e = np.einsum("bii->b", As)
products_e = np.einsum("bij,bjk->bik", As, Bs)
time_e = time.perf_counter() - start

start = time.perf_counter()
traces_l = np.zeros(100)
products_l = np.zeros((100, 4, 4))
for b in range(100):
    traces_l[b] = np.trace(As[b])
    products_l[b] = As[b] @ Bs[b]
time_l = time.perf_counter() - start

print("batch traces agree with a loop:  ", np.allclose(traces_e, traces_l))
print("batch products agree with a loop:", np.allclose(products_e, products_l))
print(f"time: einsum {time_e * 1000:.3f} ms, loop {time_l * 1000:.3f} ms")

print("\nPart 3: Kronecker product")
A = rng.normal(size=(3, 4))
B = rng.normal(size=(2, 5))
X = rng.normal(size=(4, 5))
print("(A X B^T).reshape(-1) == kron(A, B) @ X.reshape(-1):",
      np.allclose((A @ X @ B.T).reshape(-1), np.kron(A, B) @ X.reshape(-1)))
print("numbers stored in kron(A, B) for 100x100 A and B:", 100 ** 4)

print("\nPart 4: rank-one tensors")


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


u = rng.normal(size=3)
v = rng.normal(size=4)
w = rng.normal(size=5)
T = rank_one(u, v, w)
ok = True
for k in range(5):
    if not np.allclose(T[:, :, k], w[k] * np.outer(u, v)):
        ok = False
print("every slice T[:, :, k] equals w_k * u v^T:", ok)

print("""
Answers:
1. Both unfoldings have rank 2. In T.reshape(2, 12) the two rows are 0..11 and 12..23; they are not
   multiples of each other, so the rank is 2. In the 3x8 unfolding every row is (first row) + 4j*(1,...,1),
   so all rows lie in the span of two vectors, and the rank is again 2.
2. The batched einsum gives the same traces and products as the Python loop, and it is faster,
   because the loop over the batch runs in compiled code instead of in Python.
3. kron(A, B) for A, B in R^{100x100} is a 10000 x 10000 matrix: 10^8 numbers (about 800 MB).
   Computing A X B^T directly needs two matrix products of 100 x 100 matrices, about 2 * 100^3 = 2 * 10^6
   multiplications and no extra memory, while multiplying by kron(A, B) needs 10^8 multiplications.
4. T[:, :, k] = w_k u v^T, so every slice is a multiple of the same rank-one matrix u v^T.
""")
