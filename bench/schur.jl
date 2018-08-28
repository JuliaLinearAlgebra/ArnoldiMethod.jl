using ArnoldiMethod, LinearAlgebra
using BenchmarkTools

function bench_schur(n = 30)
    H = triu(rand(n, n), -1)
    Q = Matrix{Float64}(I, n, n)
    a = @benchmark ArnoldiMethod.local_schurfact!(HH, 1, $n, QQ) setup = (HH = copy($H); QQ = copy($Q))
    b = @benchmark eigvals(HH) setup = (HH = copy($H))
    a, b
end
