using Base.Test

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!

@testset "Implicit restart" begin

    min = 5
    max = 8

    A = rand(Complex128, 32, 32)

    n = size(A, 1)

    arnoldi = initialize(Complex128, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    implicit_restart!(arnoldi, min, max)
    V = arnoldi.V
    H = arnoldi.H
    @test vecnorm(V[:, 1 : min-1]' * V[:, 1 : min-1] - eye(min-1)) < 1e-13 #min-1?
    @test vecnorm(V[:, 1 : min-1]' * A * V[:, 1 : min-1] - H[1 : min-1, 1 : min-1]) < 1e-13
    @test vecnorm(A * V[:, 1 : min-1] - V[:, 1 : min] * H[1 : min, 1 : min-1]) < 1e-13

end
