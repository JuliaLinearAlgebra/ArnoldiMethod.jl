using Base.Test

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!, restarted_arnoldi_2

@testset "Restarted Arnoldi 2" begin
    # n = 5
    criterion = 1e-2
    # H = triu(rand(Complex128, n+1,n), -1)
    # H_new = copy(H)

    # rotations = ListOfRotations(eltype(H),n-1)

    A = rand(Complex128, 8, 8)
    
    # λs = sort!(eigvals(A), by = abs, rev = true)
    λ, m = restarted_arnoldi_2(A, 5, 7, criterion)
    # Base.showarray(STDOUT,[λs, λ],false)
    # @test λ ≈ λs[1]
    @test norm(A * m - λ * m) < criterion

end
