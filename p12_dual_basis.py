import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)


def inverse(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    M = np.hstack([A, np.eye(n)])
    for c in range(n):
        best = c + np.argmax(np.abs(M[c:, c]))
        if abs(M[best, c]) < 1e-12:
            raise np.linalg.LinAlgError("matrix is singular")
        M[[c, best]] = M[[best, c]]
        M[c] = M[c] / M[c, c]
        for i in range(n):
            if i != c:
                M[i] = M[i] - M[i, c] * M[c]
    return M[:, n:]


def dual_basis(P):
    return inverse(P)


print("Part 1: random 5x5 matrix P")
P = rng.normal(size=(5, 5))
Rows = dual_basis(P)
print("r_i b_j is the identity:", np.allclose(Rows @ P, np.eye(5)))
x = rng.normal(size=5)
y = np.zeros(5)
for i in range(5):
    y = y + (Rows[i] @ x) * P[:, i]
print("x = sum (r_i x) b_i:", np.allclose(x, y))
print("same as np.linalg.inv:", np.allclose(Rows, np.linalg.inv(P)))


def vandermonde(t, d):
    V = np.zeros((len(t), d + 1))
    for k in range(len(t)):
        for j in range(d + 1):
            V[k, j] = t[k] ** j
    return V


def evaluate(coef, x):
    result = np.zeros_like(x, dtype=float)
    for j in range(len(coef)):
        result = result + coef[j] * x ** j
    return result


print("\nPart 2: Lagrange polynomials for the nodes -1, 0, 1")
t = np.array([-1.0, 0.0, 1.0])
L = inverse(vandermonde(t, 2))
for j in range(3):
    print(f"L_{int(t[j])}: coefficients (1, x, x^2) =", L[:, j])
print("expected: x(x-1)/2 -> [0, -0.5, 0.5],  1 - x^2 -> [1, 0, -1],  x(x+1)/2 -> [0, 0.5, 0.5]")


def interpolate(f, t):
    L = inverse(vandermonde(t, len(t) - 1))
    return L @ f(t)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

f1 = lambda s: np.sin(3 * s)
t1 = np.linspace(-1, 1, 9)
xs = np.linspace(-1, 1, 400)
axes[0].plot(xs, f1(xs), label="f(t) = sin(3t)")
axes[0].plot(xs, evaluate(interpolate(f1, t1), xs), "--", label="interpolant p")
axes[0].plot(t1, f1(t1), "o", label="nodes")
axes[0].set_title("9 nodes on [-1, 1]")
axes[0].legend()

f2 = lambda s: 1 / (1 + s ** 2)
t2 = np.linspace(-5, 5, 21)
xs = np.linspace(-5, 5, 1000)
axes[1].plot(xs, f2(xs), label="f(t) = 1/(1+t^2)")
axes[1].plot(xs, evaluate(interpolate(f2, t2), xs), "--", label="interpolant p")
axes[1].plot(t2, f2(t2), "o", label="nodes")
axes[1].set_ylim(-2, 2)
axes[1].set_title("21 nodes on [-5, 5]: Runge's phenomenon")
axes[1].legend()
plt.tight_layout()
plt.savefig("p12_interpolation.png", dpi=120)

err1 = np.max(np.abs(f1(np.linspace(-1, 1, 400)) - evaluate(interpolate(f1, t1), np.linspace(-1, 1, 400))))
err2 = np.max(np.abs(f2(xs) - evaluate(interpolate(f2, t2), xs)))
print("\nPart 3: interpolation")
print("max error, sin(3t), 9 nodes:", err1)
print("max error, 1/(1+t^2), 21 nodes:", err2)

print("""
Answers:
1. The columns b_j of P and the rows r_i of P^{-1} satisfy r_i b_j = (P^{-1} P)_{ij} = delta_ij,
   so the rows of P^{-1} are the dual basis. Applying them to x gives P^{-1} x, the coordinates of x.
2. Row k of V is the evaluation functional p -> p(t_k). The coefficient vector of L_j must satisfy
   p(t_k) = delta_jk, i.e. V L_j = e_j, so L_j = V^{-1} e_j is the j-th column of V^{-1}.
   In other words, the Lagrange polynomials form the basis whose dual basis is the evaluations.
3. For sin(3t) on 9 nodes the interpolant is almost indistinguishable from f.
   For 1/(1+t^2) on 21 equally spaced nodes in [-5, 5], the polynomial of degree 20 oscillates
   wildly near the ends of the interval: it matches f at the nodes, but between the outer nodes
   the error is huge and grows when more equally spaced nodes are added (Runge's phenomenon).
""")
