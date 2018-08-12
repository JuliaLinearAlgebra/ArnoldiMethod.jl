# Does not run in 0.7
using BenchmarkTools

mymatrix(n) = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.001, n-1))

function bencheigs()
    A = mymatrix(6000)
    @benchmark eigs(B, nev=10, which=:SM) setup = (B = $A)
end
