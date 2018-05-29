using Base.Test

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!, restarted_arnoldi_2

@testset "Restarted Arnoldi 2" begin
    # n = 5
    criterion = 1e-5
    # H = triu(rand(Complex128, n+1,n), -1)
    # H_new = copy(H)

    # rotations = ListOfRotations(eltype(H),n-1)

    # n = 1000
    # A = spdiagm((fill(-1.01 + 0im, n - 1), fill(2.0 + 0im, n), fill(-1.0 + 0im, n - 1)), (-1,0,1))

    A = rand(Complex128, 32, 32)
    
    # λs = sort!(eigvals(A), by = abs, rev = true)
    λ, x = restarted_arnoldi_2(A, 5, 8, criterion)

    # @test vecnorm(V[:, 1 : 5]' * V[:, 1 : 5] - eye(5)) < 1e-10
    # @test vecnorm(V[:, 1 : 5]' * A * V[:, 1 : 5] - H[1 : 5, 1 : 5]) < 1e-11
    # @test vecnorm(A * V[:, 1 : 5] - V[:, 1 : 6] * H[1 : 6, 1 : 5]) < 1e-10

    @show norm(x) ≈ 1
    @test norm(A * x - λ * x) < criterion
end
