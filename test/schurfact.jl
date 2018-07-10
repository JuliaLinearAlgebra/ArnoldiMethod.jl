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
    
    H1 = triu(rand(10,10), -1)
    @show eigvals(H)
    H_copy = copy(H1)
    Q1 = eye(10)
    schurfact!(H1, Q1, 1, 10)
    display(H1)
    @test vecnorm(Q1*H1*Q1' - H_copy) < 1e-8

    H2 = [ 31.3265       -4.09316       1.95146      -0.770332     -0.220247      0.451145;
    0.133979     30.4833       -2.25368       0.70086      -0.456425      0.255826;   
    0.0           0.255179     29.8498       -1.13323       0.21986      -0.157696;      
    3.38813e-21   8.67362e-19   0.410006     29.1263       -0.239704      0.0348652;     
    2.97702e-22   0.0          -1.73472e-18   0.447611     29.0287       -0.216972;      
    0.0           5.36667e-22   0.0           1.73472e-18   0.549446     29.2115]

    @show eigvals(H2)
    H_copy2 = copy(H2)
    Q2 = eye(6)
    schurfact!(H2, Q2, 1, 6)
    # display(H2)
    @test vecnorm(Q2*H2*Q2' - H_copy2) < 1e-8

end
