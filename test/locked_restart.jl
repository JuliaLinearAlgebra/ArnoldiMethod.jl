using Test
using LinearAlgebra

using IRAM: restarted_arnoldi

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
    min, max = 25, 35
    for T in (Float64, ComplexF64)
        A = matrix_with_three_clusters(T, 100)
        ε = 1e-6

        # Get the Arnoldi relation after seven restarts.
        partial_schur = restarted_arnoldi(A, min, max, min, eps(real(T)), 20)

        R, Q, k = partial_schur.R, partial_schur.Q, partial_schur.k

        # Testing the Arnoldi relation AV = VH
        @test norm(Q[:, 1 : k]' * A * Q[:, 1 : k] - R[1 : k, 1 : k]) < ε
        @test norm(Q[:, 1 : k]' * Q[:, 1 : k] - I) < ε
        @test norm(A * Q[:, 1 : k] - Q[:, 1 : k + 1] * R[1 : k + 1, 1 : k]) < ε

        @test abs(R[4, 3]) ≤ ε
        @test abs(R[21, 20]) ≤ ε

        V₁ = view(partial_schur.Q, :, 1 : 3)
        V₂ = view(partial_schur.Q, :, 4 : 20)

        # Compute the first 3 approx eigenvalues and eigenvectors.
        Λ₁, Y₁ = eigen(R[1:3, 1:3])
        X₁ = V₁ * Y₁

        # Look at the residuals.
        for i = 1 : length(Λ₁)
            r = norm(A * X₁[:, i] - Λ₁[i] * X₁[:, i])
            @test r ≤ ε
        end

        # The eigenvalues 4 .. 20 are the dominant eigenvalues of the matrix (I-V1V1')A(I-V1V1')
        Λ₂, Y₂ = eigen(R[4:20, 4:20])
        X₂ = V₂ * Y₂

        # Look at the residuals (I - V₁V₁')A. Note that we repeat the orthogonalization to avoid
        # numerical errors! In exact arithmetic (I - V₁V₁')² = (I - V₁V₁')
        for i = 1 : length(Λ₂)
            r = A * X₂[:, i] - Λ₂[i] * X₂[:, i]
            r .-= V₁ * (V₁' * r)
            r .-= V₁ * (V₁' * r)
            @test norm(r) ≤ ε
        end
    end
        # Test the orthonormality of V
        # @test norm(Q[:,1:min]'*Q[:,1:min] - I) < 1e-6
        # S =  Q[:,1:min]'*Q[:,1:min] - I
        # for i = 1:min
        #     for j = 1:min
        #         if abs(S[i,j]) > 1e-9
        #             @show (i,j)
        #             @show S[i,j]
        #         end
        #     end
        # end

        # Test the orthonormality of Q
        # @test norm(Q[:,1:k]'*Q[:,1:k] - I) < 1e-6

        # Test that R is upper triangular
        # @test norm(triu(R[1:k,1:k])-R[1:k,1:k]) < 1e-4

        # for i = 1:min
        #     for j = 1:min
        #         if abs(R[i,j]) > 1e-9
        #             @show (i,j)
        #         end
        #     end
        # end

        # Test the partial Schur decomposition relation AQ = QR
        # @test norm(A*Q[:,1:k] - Q[:,1:k]*R[1:k,1:k]) < 1e-6
end


# Fails, as A has complex eigenvals
# @testset "Special case" begin
#    ε = 1e-6
#    min, max = 25, 35
#    A = triu(rand(Float64, 100, 100))
#    for i = 1 : 100
#        A[i,i] = i
#    end
#     # A = matrix_with_three_clusters(Float64, 100)
#    A[25:26,25:26] = [0.0 26.0; -26.0 0.0] 
# #    A[1:2,1:2] = [1.0 22.0; -22.0 1.0]
# #    A[1:2,1:2] = [2.0 1.0; 1.0 1.0]
# #    display(A)
#    partial_schur = restarted_arnoldi(A, min, max, 5, eps(Float64), 100)
#    # partial_schur = restarted_arnoldi(A, min, max, min, eps(real(T)), 10)

#    R, Q, k = partial_schur.R, partial_schur.Q, partial_schur.k
#    @show k
#    # Testing the Arnoldi relation AV = VH
#    @test norm(Q[:, 1 : k]' * A * Q[:, 1 : k] - R[1 : k, 1 : k]) < ε
#    @test norm(Q[:, 1 : k]' * Q[:, 1 : k] - I) < ε
#    @test norm(A * Q[:, 1 : k] - Q[:, 1 : k + 1] * R[1 : k + 1, 1 : k]) < ε

#    # @test abs(R[4, 3]) ≤ ε
#    # @test abs(R[21, 20]) ≤ ε

#    V₁ = view(partial_schur.Q, :, 1 : k)
#    # V₂ = view(partial_schur.Q, :, 4 : 20)

#    # Compute the first k approx eigenvalues and eigenvectors.
#    Λ₁, Y₁ = eigen(R[1:k, 1:k])
#    X₁ = V₁ * Y₁

#    # Look at the residuals.
#    for i = 1 : length(Λ₁)
#        r = norm(A * X₁[:, i] - Λ₁[i] * X₁[:, i])
#        @test r ≤ ε
#    end
# end

