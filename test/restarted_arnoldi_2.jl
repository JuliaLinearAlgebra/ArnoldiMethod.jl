using Base.Test

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!, restarted_arnoldi_2

@testset "Restarted Arnoldi 2" begin
    criterion = 1e-5
    max_restarts = 50

    A = rand(Complex128, 32, 32)
    
    λ, x = restarted_arnoldi_2(A, 5, 6, criterion, 50)

    # @show norm(x) ≈ 1
    @test norm(A * x - λ * x) < criterion
end
