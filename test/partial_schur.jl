using Test

using ArnoldiMethod: partialschur, vtype, eigenvalues, SR
using LinearAlgebra

@testset "Zero eigenvalues & low-rank matrices" begin
    # Construct a rank-3 matrix, then with 100% change it'll find the first
    # three eigenvectors in 3 iterations. From then on, all random vectors
    # form a basis for the multiple eigenvalue -- for now we just test that
    # the stopping criterion works for these zero eigenvalues, we don't take
    # into account that it finds the complete eigenspace of the 0 eigenvalue.
    # (we have too few basis vectors for it anyways)
    for T in (Float64, ComplexF64, BigFloat, Complex{BigFloat})
        A = rand(T,10, 3)

        # Rank 3 matrix.
        B = A * A' 

        schur, history = partialschur(B, nev = 5, mindim = 5, maxdim = 7, tol = eps())

        @test history.converged
        @test history.mvproducts == 7
        @test norm(schur.Q'schur.Q - I) < 100eps(real(T))
        @test norm(B * schur.Q - schur.Q * schur.R) < 100eps(real(T))
        @test norm(diag(schur.R)[4:5]) < 100eps(real(T))
    end
end

@testset "Right number type" begin
    A = [rand(Bool) ? 1 : 0 for i=1:10, j=1:10]
    @inferred partialschur(A, nev = 2, mindim = 3, maxdim = 8)
    @test vtype(A) == Float64
end

@testset "Find all eigenvalues of a small matrix" begin
    A = rand(3, 3)
    schur, history = partialschur(A)
    @test history.converged
    @test history.mvproducts == 3
end

@testset "Incorrect input" begin
    A = rand(6, 6)

    @test_throws DimensionMismatch partialschur(rand(4, 3))
    @test_throws ArgumentError partialschur(A, mindim = 5, maxdim = 3)
    @test_throws ArgumentError partialschur(A, nev = 5, mindim = 3)
    @test_throws ArgumentError partialschur(A, nev = 5, maxdim = 3)
    @test_throws ArgumentError partialschur(A, nev = 10)
end

@testset "Target non-dominant eigenvalues" begin
    # Dominant eigenvalues 50, 51, 52, 53, but we target the smallest real part
    A = Diagonal([1:0.1:10; 50:53])
    S, hist = partialschur(A, which = SR())
    @test all(x -> real(x) ≤ 10, eigenvalues(S.R))
end

@testset "Repeated eigenvalues" begin
    # Repeated eigenvalues have somewhat irregular convergence behavior. In
    # the example below the dominant eigenvalue 10.0 is repeated, and typically
    # converges first, but the next multiple may only start to converge after
    # 9.99, 9.98, 9.97 are converged. We have no purging, so eigenvalues are
    # locked as they converge, meaning that we may not always find all largest
    # magnitude eigenvalues. This test merely checks if we do get a correct
    # Schur decomposition -- in the past purging was implemented incorrectly,
    # destroying the partial schur decomposition.
    A = Diagonal([1:0.1:9; 9.97; 9.98; 9.99; 10.0; 10.0; 10.0])

    schur, history = partialschur(A, nev=5, maxdim=20, tol=1e-12)
    @test history.converged
    @test norm(schur.Q'schur.Q - I) < 100 * eps(Float64)
    @test norm(A * schur.Q - schur.Q * schur.R) < size(A, 1) * 1e-12
end

