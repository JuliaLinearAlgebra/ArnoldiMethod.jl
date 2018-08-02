# Partial Schur decomposition

Internally the Arnoldi method first builds a partial Schur decomposition of a
matrix $A$, and only then transforms this to an eigendecomposition. In some
cases one might wish to just have the Schur vectors, because they form a stable,
orthonormal basis for the eigenspace.

By default IRAM.jl returns a partial Schur decomposition

```math
AQ = QR
```
where $Q$ is orthonormal of size $n \times \texttt{nev}$ and $R$ is upper 
triangular of size $\texttt{nev} \times \texttt{nev}.$ In real arithmetic $R$
is quasi upper triangular, with $2 \times 2$ blocks on the diagonal 
corresponding to conjugate complex-valued eigenpairs.

Via [some function] one can transform the partial Schur decomposition to an
eigendecomposition. Suppose $RY = YS$ is the eigendecomposition of $R$, then the
eigendecomposition of $A$ is

```math
A(QY) = (QY)S
```

Note that $Y$ is upper triangular as well, so the product $QY$ can be computed
efficiently and even in-place.