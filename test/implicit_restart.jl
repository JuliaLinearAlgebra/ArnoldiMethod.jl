using Test

using IRAM: implicit_restart!, initialize, iterate_arnoldi!

@testset "Implicit restart" begin

    for T in (Float64, Complex128)

        n = 20
        A = sprand(T, n, n, 5 / n) + I
        min, max = 5, 8
        h = Vector{T}(max)

        arnoldi = initialize(T, n, max)
        V, H = arnoldi.V, arnoldi.H
        iterate_arnoldi!(A, arnoldi, 1 : max, h)
        λs = sort!(eigvals(view(H, 1:max, 1:max)), by = abs, rev = true)

        m = implicit_restart!(arnoldi, λs, min, max)

        @test vecnorm(V[:, 1 : m]' * V[:, 1 : m] - eye(m)) < 1e-13
        @test vecnorm(V[:, 1 : m]' * A * V[:, 1 : m] - H[1 : m, 1 : m]) < 1e-13
        @test vecnorm(A * V[:, 1 : m] - V[:, 1 : m + 1] * H[1 : m + 1, 1 : m]) < 1e-13
    end
end
