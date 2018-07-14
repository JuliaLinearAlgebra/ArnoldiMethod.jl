using IRAM, LinearAlgebra
using BenchmarkTools

function bench_schur()
    H = triu(rand(30, 30), -1)
    Q = Matrix{Float64}(I, 30, 30)
    a = @benchmark IRAM._local_schurfact!(HH, 1, 30) setup = (HH = copy($H))
    b = @benchmark IRAM.local_schurfact!(HH, QQ, 1, 30) setup = (HH = copy($H); QQ = copy($Q))
    c = @benchmark IRAM._local_schurfact!(HH, 1, 30, QQ) setup = (HH = copy($H); QQ = copy($Q))
    d = @benchmark eigvals(HH) setup = (HH = copy($H))
    a, b, c, d
end
