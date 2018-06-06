using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, restarted_arnoldi

@testset "Restarted Arnoldi" begin
    criterion = 1e-10
    max_restarts = 50

    # A = full(sprand(Complex128,10000, 10000, 5 / 10000))
    A = full(sprand(Complex128,1000, 1000, 5 / 1000))

    λ, x = restarted_arnoldi(A, 10, 50, criterion, 50)

    # @show norm(A * x - λ * x) < criterion
    @test norm(A * x - λ * x) < criterion


    # criterion = 1e-5
    # max_restarts = 50

    # A = rand(Complex128, 32, 32)
    
    # λ, x = restarted_arnoldi_2(A, 5, 6, criterion, 50)

    # # @show norm(x) ≈ 1
    # @test norm(A * x - λ * x) < criterion
end
