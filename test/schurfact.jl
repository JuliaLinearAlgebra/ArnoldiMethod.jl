using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, restarted_arnoldi, eigvalues, schurfact!

@testset "Schur factorization" begin

    # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
    H = [1.0 2.0; 3.0 4.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
    H = [1.0 2.0; 0.0 4.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with conjugate eigenvalues
    H = [1.0 4.0; -5.0 3.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8
    
end
