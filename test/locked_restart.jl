using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, restarted_arnoldi

@testset "Locked restart" begin

    min = 25
    max = 50

    A = triu(rand(Complex128, 100, 100))
    for i = 1 : 3
        A[i,i] = 100 + i
    end
    for i = 4 : 20
        A[i,i] = 29 + 0.05*i
    end
    for i = 21 : 100
        A[i,i] = 1 + 0.05*i
    end

    n = size(A, 1)

    arnoldi = restarted_arnoldi(A, min, max, 1e-6, 100)

    return arnoldi

    arnoldi = initialize(Complex128, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    arnoldi, dim, i = implicit_restart!(arnoldi, min, max)

    λs, xs = eig(view(arnoldi.H, 1 : dim, 1 : dim))
    perm = sortperm(λs, by = abs, rev = true)
    λs = λs[perm]
    xs = view(xs, :, perm)

    V = arnoldi.V
    H = arnoldi.H

    v = view(arnoldi.V, :, 1 : dim) * xs[:,1]

    @show i,dim
    for n = 1:dim
        @show norm(A * view(arnoldi.V, :, 1 : dim) * xs[:,n] - λs[n] * view(arnoldi.V, :, 1 : dim) * xs[:,n])
    end
    # @test norm(A * v - λs[1] * v) < 1e-10

    # @test vecnorm(V[:, 1 : min-1]' * V[:, 1 : min-1] - eye(min-1)) < 1e-13
    # @test vecnorm(V[:, 1 : min-1]' * A * V[:, 1 : min-1] - H[1 : min-1, 1 : min-1]) < 1e-13
    # @test vecnorm(A * V[:, 1 : min-1] - V[:, 1 : min] * H[1 : min, 1 : min-1]) < 1e-13


    # criterion = 1e-10
    # max_restarts = 50

    # A = full(sprand(Complex128,10000, 10000, 5 / 10000))

    # λ, x = restarted_arnoldi_2(A, 10, 50, criterion, 4750)

    # # @show norm(A * x - λ * x) < criterion
    # @test norm(A * x - λ * x) < criterion

end
