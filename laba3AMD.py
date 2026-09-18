import numpy as np

n = 3
T = [3, 4, 1, 2]

A = np.random.rand(n, n, n, n)
B = np.random.rand(n, n, n, n)

def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p_ in enumerate(perm):
        inv[p_ - 1] = i + 1
    return inv

T_inv = inverse_permutation(T)
A_T = np.transpose(A, [t - 1 for t in T_inv])

D = np.zeros((n, n, n, n))
for l1 in range(n):
    for l2 in range(n):
        for m1 in range(n):
            for m2 in range(n):
                s = 0.0
                for c1 in range(n):
                    for c2 in range(n):
                        s += A[l1, l2, c1, c2] * B[c1, c2, m1, m2]
                D[l1, l2, m1, m2] = s
E = np.zeros((n, n, n, n))
for c1 in range(n):
    for c2 in range(n):
        E[c1, c2, c1, c2] = 1.0

F = np.zeros((n, n, n, n))
for l1 in range(n):
    for l2 in range(n):
        for m1 in range(n):
            for m2 in range(n):
                s = 0.0
                for c1 in range(n):
                    for c2 in range(n):
                        s += E[l1, l2, c1, c2] * B[c1, c2, m1, m2]
                F[l1, l2, m1, m2] = s

# проверяем чтобы F совпадало с B 
err = np.max(np.abs(F - B))
print(f"Проверка F = B: max|F - B| = {err:.3e}")

np.set_printoptions(precision=4, suppress=True, linewidth=140)

print("\nA_T (транспонированная по T):")
print(A_T)

print("\nD = (λ,μ)-свёрнутое произведение A и B:")
print(D)

print("\nE(λ,μ) — (λ,μ)-единичная матрица:")
print(E)

print("\nF = (λ,μ)(E(λ,μ) B):")
print(F)