using IRAM, LinearAlgebra
using BenchmarkTools

function bench_schur(n = 30)
    H = triu(rand(n, n), -1)
    Q = Matrix{Float64}(I, n, n)
    a = @benchmark IRAM._local_schurfact!(HH, 1, $n) setup = (HH = copy($H))
    b = @benchmark IRAM._local_schurfact!(HH, 1, $n, QQ) setup = (HH = copy($H); QQ = copy($Q))
    c = @benchmark IRAM.local_schurfact!(HH, QQ, 1, $n) setup = (HH = copy($H); QQ = copy($Q))
    d = @benchmark eigvals(HH) setup = (HH = copy($H))
    a, b, c, d
end
