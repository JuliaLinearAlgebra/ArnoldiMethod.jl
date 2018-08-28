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
    A = sprand(n, n, .1) + I

    arnoldi = Arnoldi{Float64}(n, max)
    reinitialize!(arnoldi)
    V, H = arnoldi.V, arnoldi.H

    # Do a few iterations
    iterate_arnoldi!(A, arnoldi, 1:3)
    @test A * V[:,1:3] ≈ V[:,1:4] * H[1:4,1:3]
    @test norm(V[:,1:4]' * V[:,1:4] - I) < 1e-10

    # Do the rest of the iterations.
    iterate_arnoldi!(A, arnoldi, 4:max)
    @test A * V[:,1:max] ≈ V * H
    @test norm(V' * V - I) < 1e-10
end

@testset "Invariant subspace" begin
    # Generate a block-diagonal matrix A
    A = [rand(4,4)  zeros(4,4); 
         zeros(4,4) rand(4,4)]

    # and an initial vector [1; 0; ... 0]
    vh = Arnoldi{Float64}(8, 5)
    V, H = vh.V, vh.H
    V[:,1] .= 0.0
    V[1,1] = 1.0

    # Then {v, Av, A²v, A³v}
    # is an invariant subspace
    iterate_arnoldi!(A, vh, 1:5)

    @test norm(V' * V - I) < 1e-10
    @test iszero(H[5, 4])
end