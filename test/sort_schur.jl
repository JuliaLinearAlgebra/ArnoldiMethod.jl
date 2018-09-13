using Test
using LinearAlgebra
using StaticArrays
using ArnoldiMethod: swap11!, swap12!, swap21!, swap22!, rotate_right!, eigenvalues,
                     sylvsystem, CompletePivoting

# These tests are only necessary in real arithmetic, but why not do complex for completeness

@testset "Reordering the Schur form 1 ↔ 1 ($T)" for T in (Float64, ComplexF64) 
    Q1 = Matrix{T}(I, 2, 2)
    R1 = triu(rand(T, 2, 2))
    Q2 = copy(Q1)
    R2 = copy(R1)

    swap11!(R2, 1, Q2)

    @test R2[1,1] ≈ R1[2,2]
    @test R1[1,1] ≈ R2[2,2]
    @test R1 * Q2 ≈ Q2 * R2
end

@testset "Reordering the Schur form 1 ↔ 2 ($T)" for T in (Float64, ComplexF64) 
    # x x x
    # . x x
    # . x x
    Q1 = Matrix{T}(I, 3, 3)
    R1 = triu(rand(T, 3, 3))
    R1[3,2] = rand(T)

    Q2 = copy(Q1)
    R2 = copy(R1)

    swap12!(R2, 1, Q2)

    @test iszero(R2[3,1])
    @test iszero(R2[3,2])
    @test R1[1,1] ≈ R2[3,3]
    @test sort!(eigvals(R1[2:3,2:3]), by = reim) ≈ sort!(eigvals(R2[1:2,1:2]), by = reim)
    @test R1 * Q2 ≈ Q2 * R2
end

@testset "Reordering the Schur form 2 ↔ 1 ($T)" for T in (Float64, ComplexF64)
    # x x x
    # x x x
    # . . x
    Q1 = Matrix{T}(I, 3, 3)
    R1 = triu(rand(T, 3, 3))
    R1[2,1] = rand(T)

    Q2 = copy(Q1)
    R2 = copy(R1)

    swap21!(R2, 1, Q2)

    @test iszero(R2[2,1])
    @test iszero(R2[3,1])
    @test R1[3,3] ≈ R2[1,1]
    @test sort!(eigvals(R1[1:2,1:2]), by = reim) ≈ sort!(eigvals(R2[2:3,2:3]), by = reim)
    @test R1 * Q2 ≈ Q2 * R2
end

@testset "Reordering the Schur form 2 ↔ 2 ($T)" for T in (Float64, ComplexF64)
    # x x x x
    # x x x x
    # . . x x
    # . . x x
    Q1 = Matrix{T}(I, 4, 4)
    R1 = triu(rand(T, 4, 4))
    R1[2,1] = rand(T)
    R1[4,3] = rand(T)

    Q2 = copy(Q1)
    R2 = copy(R1)

    swap22!(R2, 1, Q2)

    @test iszero(R2[3,1])
    @test iszero(R2[4,1])
    @test iszero(R2[3,2])
    @test iszero(R2[4,2])
    @test sort!(eigvals(R1[1:2,1:2]), by = reim) ≈ sort!(eigvals(R2[3:4,3:4]), by = reim)
    @test sort!(eigvals(R1[3:4,3:4]), by = reim) ≈ sort!(eigvals(R2[1:2,1:2]), by = reim)
    @test R1 * Q2 ≈ Q2 * R2
end

@testset "Rotation right with single eigenvalues" begin
    # 10 × 10 quasi upper triangular matrix with 2 × 2 block on R[4:5,4:5]
    R = triu(rand(10, 10))
    Q = Matrix(1.0I, 10, 10)
    R[4,5] = -2.0; R[5,4] = 2.0

    λ_before = eigenvalues(R)

    # Rotate the last block right before the first one.
    R_after = copy(R)
    rotate_right!(R_after, 1, 10, Q)
    λ_after = eigenvalues(R_after)

    # Test whether Q is more or less a similarity transform
    @test opnorm(R - Q * R_after * Q', 1) < 10eps() * opnorm(R, 1)

    # Test orthonormality of Q
    @test norm(Q'Q - I) < 10eps()

    # Middle guys have been rotated
    for (i, j) = zip(1:10, circshift(1:10, -1))
        @test λ_before[i] ≈ λ_after[j]
    end
end

@testset "Rotation right with two conjugate pairs" begin
    # 10 × 10 quasi upper triangular matrix with 2 × 2 blocks on R[2:3,2:3] and R[6:7,6:7]
    R = triu(rand(10, 10))
    Q = Matrix(1.0I, 10, 10)
    R[3,2] = -2.0; R[2,3] = 2.0
    R[7,6] = 3.0; R[6,7] = -2.0

    λ_before = eigenvalues(R)

    # Rotate the last block right before the first one.
    R_after = copy(R)
    rotate_right!(R_after, 3, 6, Q)

    λ_after = eigenvalues(R_after)

    # Test whether Q is more or less a similarity transform
    @test opnorm(R - Q * R_after * Q', 1) < 10eps() * opnorm(R, 1)

    # Test orthonormality of Q
    @test norm(Q'Q - I) < 10eps()

    # First eigenvalue should be exactly equal
    @test λ_before[1] == λ_after[1]

    # Middle guys have been rotated
    for (i, j) = zip(2:7, circshift(2:7, -2))
        @test λ_before[i] ≈ λ_after[j]
    end

    # Last eigenvalues should be exactly equal
    @test λ_before[8:10] == λ_after[8:10]
end

@testset "Rotation right with one 2 × 2 block on the right" begin
    # 10 × 10 quasi upper triangular matrix with 2 × 2 block on R[6:7,6:7]
    R = triu(rand(10, 10))
    Q = Matrix(1.0I, 10, 10)
    R[6,7] = -2.0; R[7,6] = 2.0

    λ_before = eigenvalues(R)

    # Rotate the last block right before the first one.
    R_after = copy(R)
    rotate_right!(R_after, 2, 6, Q)
    λ_after = eigenvalues(R_after)

    # Test whether Q is more or less a similarity transform
    @test opnorm(R - Q * R_after * Q', 1) < 10eps() * opnorm(R, 1)

    # Test orthonormality of Q
    @test norm(Q'Q - I) < 10eps()

    # First eigenvalue should be exactly equal
    @test λ_before[1] == λ_after[1]

    # Middle guys have been rotated
    for (i, j) = zip(2:7, circshift(2:7, -2))
        @test λ_before[i] ≈ λ_after[j]
    end

    # Last eigenvalues should be exactly equal
    for i = 8:10
        @test λ_before[i] == λ_after[i]
    end

end

@testset "Rotation right with one 2 × 2 block on the left" begin
    # 10 × 10 quasi upper triangular matrix with 2 × 2 block on R[2:3,2:3]
    R = triu(rand(10, 10))
    Q = Matrix(1.0I, 10, 10)
    R[2,3] = -2.0; R[2,3] = 2.0

    λ_before = eigenvalues(R)

    # Rotate the last block right before the first one.
    R_after = copy(R)
    rotate_right!(R_after, 2, 6, Q)
    λ_after = eigenvalues(R_after)

    # Test whether Q is more or less a similarity transform
    @test opnorm(R - Q * R_after * Q', 1) < 10eps() * opnorm(R, 1)

    # Test orthonormality of Q
    @test opnorm(Q'Q - I, 1) < 10 * eps()

    # First eigenvalue should be exactly equal
    @test λ_before[1] == λ_after[1]

    # Middle guys have been rotated
    for (i, j) = zip(2:6, circshift(2:6, -1))
        @test λ_before[i] ≈ λ_after[j]
    end

    # Last eigenvalues should be exactly equal
    for i = 7:10
        @test λ_before[i] == λ_after[i]
    end

end

# Stewart's example in Bai & Demmel's article
# This basically shows that a direct method is preferred over QR iterations, because of
# forward instabilities of the QR method -- also the reason why I switched away from
# Sorensen's implicit restart and convoluted locking + purging strategies that were 
# necessary exactly because of this.
@testset "Stewart's example" begin
    A(τ) = [7.0010 -87.0000  39.4000τ  22.2000τ;
            5.0000   7.0010 -12.2000τ  36.0000τ;
            0.0000   0.0000   7.0100  -11.7567 ;
            0.0000   0.0000   37.0000    7.0100 ]
    
    for τ in (1, 10, 100)
        B = A(τ)

        # Eigenvalues do not depend on τ, but we just compute them here for ease.
        λs_before = eigenvalues(B)
        swap22!(B, 1)
        λs_after = eigenvalues(B)

        # Test swapping is approximately equal
        @test abs(λs_before[1]) ≈ abs(λs_after[3])
        @test abs(λs_before[3]) ≈ abs(λs_after[1])
    end
end

# Example taken from Bai & Demmel
@testset "Small eigenvalue separation" begin
    # This should result in a very ill-conditioned Sylvester equation.
    A = [ 1.00 -100.0   400.000 -1000.000;
          0.01    1.0  1200.000   -10.000;
          0.00    0.0     1.0+eps()    -0.010;
          0.00    0.0   100.000     1.0+eps()]

    A′ = copy(A)
    Q = Matrix(1.0I, 4, 4)
    λs_before = eigenvalues(A)
    swap22!(A′, 1, Q)
    λs_after = eigenvalues(A)
    @test abs(λs_before[1]) ≈ abs(λs_after[3])
    @test abs(λs_before[3]) ≈ abs(λs_after[1])
    @test opnorm(I - Q'Q, 1) < 10eps() # we should be able to get rid of this prefactor?
    @test opnorm(A * Q - Q * A′, 1) < opnorm(A, 1) * eps()
end

@testset "Identical eigenvalues should not blow up" begin
    A = [1.0 2.0 3.0 4.0;
         0.0 1.0 5.0 6.0;
         0.0 0.0 1.0 7.0;
         0.0 0.0 0.0 1.0]
         
    A′ = copy(A)
    swap22!(A′, 1)
    @test A == A′
    swap12!(A′, 1)
    @test A == A′
    swap21!(A′, 1)
    @test A == A′
end