# Tests the Arnoldi relation AV = VH when expanding the search subspace

using Test, LinearAlgebra, SparseArrays
using ArnoldiMethod: reinitialize!, Arnoldi, iterate_arnoldi!

@testset "Initialization" begin
    arnoldi = Arnoldi{Float64}(5, 3)
    reinitialize!(arnoldi)
    @test norm(arnoldi.V[:, 1]) ≈ 1
end

@testset "Arnoldi Factorization" begin
    n = 10
    max = 6
    for T in (Float64, BigFloat)
        A = sprand(T, n, n, .1) + I

        arnoldi = Arnoldi{T}(n, max)
        reinitialize!(arnoldi)
        V, H = arnoldi.V, arnoldi.H

        # Do a few iterations
        iterate_arnoldi!(A, arnoldi, 1:3)
        @test A * V[:,1:3] ≈ V[:,1:4] * H[1:4,1:3]
        @test norm(V[:,1:4]' * V[:,1:4] - I) < sqrt(eps(T))/100

        # Do the rest of the iterations.
        iterate_arnoldi!(A, arnoldi, 4:max)
        @test A * V[:,1:max] ≈ V * H
        @test norm(V' * V - I) < sqrt(eps(T))/100
    end
end

@testset "Invariant subspace" begin
    # Generate a block-diagonal matrix A
    for T in (Float64, BigFloat)
        A = [rand(T,4,4)  zeros(T,4,4); 
            zeros(T,4,4) rand(T,4,4)]

        # and an initial vector [1; 0; ... 0]
        vh = Arnoldi{T}(8, 5)
        V, H = vh.V, vh.H
        V[:,1] .= zero(T)
        V[1,1] = one(T)

        # Then {v, Av, A²v, A³v}
        # is an invariant subspace
        iterate_arnoldi!(A, vh, 1:5)

        @test norm(V' * V - I) < sqrt(eps(T))/100
        @test iszero(H[5, 4])
    end
end