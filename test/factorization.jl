using Base.Test
using IRAM: initialize, iterate_arnoldi!, implicit_restart!, restarted_arnoldi

@testset "Initialization" begin
    arnoldi = initialize(Float64, 5, 3)
    @test norm(view(arnoldi.V, :, 1)) ≈ 1
end

@testset "Arnoldi Factorization" begin
    n = 10
    max = 6
    A = sprand(Complex128, n, n, .1) + I

    arnoldi = initialize(eltype(A), n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    V = arnoldi.V
    H = arnoldi.H

    @test A * view(V, :, 1 : max) ≈ V * H
end

@testset "Implicit restart" begin
    n = 10
    min = 3
    max = 6
    A = sprand(Complex128, n, n, .1) + I

    arnoldi = initialize(eltype(A), n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    all_ritz = sort!(eigvals(view(arnoldi.H, 1 : max, 1 : max)), by = abs, rev = true)

    implicit_restart!(arnoldi, min, max)

    restart_ritz = sort!(eigvals(view(arnoldi.H, 1 : min, 1 : min)), by = abs, rev = true)

    # Test whether the Ritz values are preserved
    @test restart_ritz ≈ view(all_ritz, 1 : min)

    # Test wether the Arnoldi decomp is preserved
    V = arnoldi.V
    H = arnoldi.H
    @test vecnorm(A * view(V, :, 1 : min) - view(V, :, 1 : min + 1) * view(H, 1 : min + 1, 1 : min)) < 1e-10
end

@testset "Integration test" begin
    my_matrix(n) = spdiagm((fill(-1.0 + 0im, n-1), fill(2.0 + 0im, n), fill(-1.1 + 0im, n-1)), (-1, 0, 1))

    A = my_matrix(100)

    λs, _ = restarted_arnoldi(A, 5, 15, 20)
    θs, _ = eigs(A, nev = 3)

    sort!(λs, by = abs, rev = true)
    sort!(θs, by = abs, rev = true)

    @test all(x -> x < 1e-5, abs.(λs[1:3] .- θs))
end