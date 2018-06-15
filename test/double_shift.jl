using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, Arnoldi, implicit_qr_step!, double_shift!

@testset "Double Shifted QR" begin
    n = 7

    H = triu(rand(n, n), -1)
    H_new = copy(H)

    # Q = eye(Complex128, max)
    # mul!(view(Q, 1 : max, 1 : m), rotations)

    rotations = ListOfRotations(eltype(H),n-1)

    λs = sort!(eigvals(H), by = abs, rev = true)
    println(λs)
    for m = n : -1 : 4
        if !(imag(λs[m]) ≈ 0)
            double_shift!(H_new, λs[m], true)
            println(sort!(eigvals(view(H_new, 1 : n-2, 1 : n-2)), by = abs, rev = true))
            # @test λs[1 : m - 2] ≈ sort!(eigvals(view(H_new, 1 : m-2, 1 : m-2)), by = abs, rev = true)
            break
        end
    end
    println(sort!(eigvals(H_new), by = abs, rev = true))
end
