using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, Arnoldi

@testset "Shifted QR" begin

    function qr_callback(arnoldi::Arnoldi, λs, m)

        # Q = eye(Complex128, max)
        # mul!(view(Q, 1 : max, 1 : m), rotations)

        # rotations = ListOfRotations(eltype(H),n-1)

        # λs = sort!(eigvals(view(arnoldi.H, 1 : n, 1 : n)), by = abs, rev = true)

        # for m = 5 : -1 : 3
            # shifted_qr_step!(H_new, λs[m], rotations)

            @test λs[1 : m - 1] ≈ sort!(eigvals(view(arnoldi.H, 1 : m-1, 1 : m-1)), by = abs, rev = true)
        # end
    end

    n = 5
    min = 3
    max = 5
    A = rand(Complex128, n,n)
    arnoldi = initialize(Complex128, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    arnoldi, dim, i = implicit_restart!(arnoldi, min, max, qr_callback)

end
