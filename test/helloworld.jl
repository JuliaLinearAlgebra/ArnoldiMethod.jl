using IRAM
using SparseArrays, LinearAlgebra

function helloworld()
    A = randn(200, 200)

    IRAM.partial_schur(A, min = 10, max = 20, tol = 1e-7, maxiter = 50)
end