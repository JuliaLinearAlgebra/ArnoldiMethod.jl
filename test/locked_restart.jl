using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, restarted_arnoldi

function matrix_with_three_clusters(n = 100)
    A = triu(rand(Complex128, n, n))
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
    A = matrix_with_three_clusters(100)
    ε = 1e-6

    # Get the Arnoldi relation after seven restarts.
    arnoldi = restarted_arnoldi(A, min, max, ε, 7)

    H, V = arnoldi.H, arnoldi.V

    @show vecnorm(A * V[:, 1 : min] - V[:, 1 : min + 1] * H[1 : min + 1, 1 : min])

    @test abs(H[4, 3]) ≤ ε
    @test abs(H[21, 20]) ≤ ε

    V₁ = view(arnoldi.V, :, 1 : 3)
    V₂ = view(arnoldi.V, :, 4 : 20)

    # Compute the first 3 approx eigenvalues and eigenvectors.
    Λ₁, Y₁ = eig(H[1:3, 1:3])
    X₁ = V₁ * Y₁

    # Look at the residuals.
    for i = 1 : length(Λ₁)
        r = norm(A * X₁[:, i] - Λ₁[i] * X₁[:, i])
        @test r ≤ ε
    end

    # The eigenvalues 4 .. 20 are the dominant eigenvalues of the matrix (I-V1V1')A(I-V1V1')
    Λ₂, Y₂ = eig(H[4:20, 4:20])
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
