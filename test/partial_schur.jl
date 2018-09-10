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

    A = rand(10, 3)

    # Rank 3 matrix.
    B = A * A' 

    schur, history = partialschur(B, nev = 5, mindim = 5, maxdim = 7, tol = eps())

    @test history.converged
    @test history.mvproducts == 7
    @test norm(schur.Q'schur.Q - I) < 100eps()
    @test norm(B * schur.Q - schur.Q * schur.R) < 100eps()
    @test norm(diag(schur.R)[4:5]) < 100eps()
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