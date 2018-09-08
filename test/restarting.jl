using LinearAlgebra
using ArnoldiMethod: lock!

function example()
    n = 10
    Q = qr(rand(n, n)).Q * Matrix(I, n, n)
    R = diagm(0 => 1:10) .+ triu(randn(n, n) .* 0.01, 1)
    R[2,3] = 1.0
    R[3,2] = -1.0

    # Some Schur decomp
    A = Q * R * Q'

    lock!(R, Q, 1, [8,9,10])
    truncate!()

    @show norm(A * Q - Q * R)

    return R

end