using BenchmarkTools

using ArnoldiMethod: partialschur, SM, LM
using LinearMaps
using LinearAlgebra
using SparseArrays
using DelimitedFiles

mymatrix(n) = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.001, n-1))

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y,x)
    mul!(M.temp, M.B, x)
    # @show typeof(y), typeof(M.A_lu), typeof(M.temp)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A,B)
    a = ShiftAndInvert(factorize(A),B,Vector{eltype(A)}(undef, size(A,1)))
    LinearMap(a, size(A,1), ismutating=true)
end

# For inverting in non-generealized case
function construct_linear_map(A)
    a = factorize(A)
    function f(y,x)
        ldiv!(y, a, x)
    end
    LinearMap{eltype(A)}(f, size(A,1))
end

function bencharnoldi()
    A = mymatrix(6000)
    A = construct_linear_map(mymatrix(6000))
    target = LM()

    @benchmark partialschur(B, mindim=11, maxdim=22, nev=10, tol=1e-10, restarts=100000, which=tar) setup = (B = $A; tar = $target)

end
