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

end
