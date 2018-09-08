using ArnoldiMethod: Arnoldi, iterate_arnoldi!, reinitialize!, partialschur,
                     Reflector, reflector!, restore_hessenberg!
using SparseArrays
using Base: OneTo
using LinearAlgebra
using Random
using BenchmarkTools

function helloworld(n = 40, m = 10)
    # A = spdiagm(0 => [range(0.1, stop=5, length=n-3); 100:102])
    # A[n,n-1] = 10.0
    # A[n-1,n] = -10.0

    A = rand(40, 40)

    partialschur(A, nev=4, mindim=4, maxdim=8)
end

function reflection_example(::Type{T} = Float64, n = 10, from = 1, to = n) where {T}

    Random.seed!(1)

    a = @benchmark LinearAlgebra.LAPACK.geev!('N', 'V', A) setup = (A = rand($T, $n, $n))
    b = @benchmark restore_hessenberg!(H, $from, $to, Q) setup = (H = triu(rand($T, $n + 1, $n)); Q = rand($T, $n, $n))

    a, b
end