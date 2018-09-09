using Test
using LinearAlgebra

using ArnoldiMethod

function matrix_with_three_clusters(T::Type, n = 100)
    A = triu(rand(T, n, n))
    for i = 1 : 3
        A[i,i] = 100 + i
    end
    for i = 4 : 20
        A[i,i] = 29 + 0.05*i
    end
    for i = 21 : n
        A[i,i] = 1 + 0.05*i
    end
    A
end

@testset "Locked restart" begin
    for T in (Float64,)
        A = matrix_with_three_clusters(T, 100)
        ε = 1e-6

        # Get the partial Schur decomposition
        S, hist = partialschur(A, nev=10, tol=ε, restarts=40)
        λs, X = partialeigen(S)

        println(hist)

        @show norm(S.Q' * S.Q - I)
        @show norm(A * X - X * Diagonal(λs))
        @show norm(A * S.Q - S.Q * S.R)

        display(S.R)
    end

        # println(history)

        # R, Q = schur_decomp.R, schur_decomp.Q
        # n = size(R,1)

        # @show norm(A * Q - Q * R) T

        # # Testing the partial Schur relation AQ = QR
        # @test norm(Q' * A * Q - R) < ε
        # @test norm(Q' * Q - I) < ε

        # # Test that the clusters have converged
        # @test abs(R[4, 3]) ≤ ε
        # @test n ≤ 20 || abs(R[21, 20]) ≤ ε

        # V₁ = view(schur_decomp.Q, :, 1 : 3)
        # V₂ = view(schur_decomp.Q, :, 4 : 20)

        # # Compute the first 3 approx eigenvalues and eigenvectors.
        # Λ₁, Y₁ = eigen(R[1:3, 1:3])
        # X₁ = V₁ * Y₁

        # # Look at the residuals.
        # for i = 1 : length(Λ₁)
        #     r = norm(A * X₁[:, i] - Λ₁[i] * X₁[:, i])
        #     @test r ≤ ε
        # end

        # # The eigenvalues 4 .. 20 are the dominant eigenvalues of the matrix (I-V1V1')A(I-V1V1')
        # Λ₂, Y₂ = eigen(R[4:20, 4:20])
        # X₂ = V₂ * Y₂

        # # Look at the residuals (I - V₁V₁')A. Note that we repeat the orthogonalization to avoid
        # # numerical errors! In exact arithmetic (I - V₁V₁')² = (I - V₁V₁')
        # for i = 1 : length(Λ₂)
        #     r = A * X₂[:, i] - Λ₂[i] * X₂[:, i]
        #     r .-= V₁ * (V₁' * r)
        #     r .-= V₁ * (V₁' * r)
        #     @test norm(r) ≤ ε
        # end
    # end

end
