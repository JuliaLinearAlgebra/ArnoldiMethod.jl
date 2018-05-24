using Base.Test

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!

@testset "Shifted QR" begin
    n = 5

    H = triu(rand(Complex128, n+1,n), -1)
    H_new = copy(H)

    # Q = eye(Complex128, max)
    # mul!(view(Q, 1 : max, 1 : m), rotations)

    rotations = ListOfRotations(eltype(H),n-1)

    λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs, rev = true)

    for m = 5 : -1 : 3
        shifted_qr_step!(H_new, λs[m], rotations)

        @test λs[1 : m - 1] ≈ sort!(eigvals(view(H_new, 1 : m-1, 1 : m-1)), by = abs, rev = true)
    end
end
