using Test, LinearAlgebra, SparseArrays
using ArnoldiMethod: Arnoldi, RitzValues, implicit_restart!, reinitialize!, iterate_arnoldi!

@testset "Implicit restart" begin

    for T in (Float64, ComplexF64)

        n = 20
        A = sprand(T, n, n, 5 / n) + I
        min, max = 5, 8

        # Construct an Arnoldi relation of size `max`
        arnoldi = Arnoldi{T}(n, max)
        reinitialize!(arnoldi)
        V, H = arnoldi.V, arnoldi.H
        Vtmp = similar(V)
        iterate_arnoldi!(A, arnoldi, 1 : max)

        # Compute some Ritz values
        ritz = RitzValues{T}(max)
        ritz.λs .= sort!(eigvals(view(H, 1:max, 1:max)), by = abs, rev = true)
        ritz.ord .= 1:max

        m = implicit_restart!(arnoldi, Vtmp, ritz, min, max)

        @test norm(V[:, 1 : m]' * V[:, 1 : m] - I) < 1e-13
        @test norm(V[:, 1 : m]' * A * V[:, 1 : m] - H[1 : m, 1 : m]) < 1e-13
        @test norm(A * V[:, 1 : m] - V[:, 1 : m + 1] * H[1 : m + 1, 1 : m]) < 1e-13
    end
end
