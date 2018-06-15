using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, Arnoldi, implicit_qr_step!, double_shift!

@testset "Double Shifted QR" begin
    n = 5

    H = triu(rand(n+1,n), -1)
    H_new = copy(H)

    # Q = eye(Complex128, max)
    # mul!(view(Q, 1 : max, 1 : m), rotations)

    rotations = ListOfRotations(eltype(H),n-1)

    λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs, rev = true)
    for m = n : -1 : 4
        if !(imag(λs[m]) ≈ 0)
            double_shift!(view(H_new,1:n,1:n), λs[m], λs[m-1], true)
            @test λs[1 : m - 2] ≈ sort!(eigvals(view(H_new, 1 : m-2, 1 : m-2)), by = abs, rev = true)
            break
        end
    end
end
