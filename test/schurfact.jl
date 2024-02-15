# Test if we can transform part of a Hessenberg matrix to Schur form.
# Here we look at some edge cases.

using Test, LinearAlgebra
using ArnoldiMethod: eigenvalues, local_schurfact!, is_offdiagonal_small, NotWanted

include("utils.jl")

@testset "Schur factorization" begin
    for T in (Float64, BigFloat)
        let
            # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
            H = [one(T) 2one(T); 3one(T) 4one(T)]
            H′ = copy(H)
            Q = Matrix{T}(I, 2, 2)

            @test local_schurfact!(H′, 1, 2, Q, eps(T), 2)
            @test norm(H * Q - Q * H′) < 10eps(T)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H′), by = realimag)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H), by = realimag)
            @test iszero(H′[2, 1])
        end

        let
            # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
            H = [one(T) 2one(T); zero(T) 4one(T)]
            H′ = copy(H)
            Q = Matrix{T}(I, 2, 2)

            @test local_schurfact!(H′, 1, 2, Q, eps(T), 2)
            @test norm(H * Q - Q * H′) < 10eps(T)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H′), by = realimag)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H), by = realimag)
            @test iszero(H′[2, 1])
        end

        let
            # 2-by-2 matrix with conjugate eigenvalues
            H = [1one(T) 4one(T); -5one(T) 3one(T)]
            H′ = copy(H)
            Q = Matrix{T}(I, 2, 2)

            @test local_schurfact!(H′, 1, 2, Q, eps(T), 2)
            @test norm(H * Q - Q * H′) < 10eps(T)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H′), by = realimag)
            @test sort!(eigenvalues(H′), by = realimag) ≈ sort!(eigvals(H), by = realimag)
        end
    end
    # Larger real matrix.
    for T in (Float64,) #randn(BigFloat,n,n) does not yet exist https://github.com/JuliaLang/julia/pull/35111
        let
            n = 10
            # Transforming a 1+i by n-i block of the matrix H into upper triangular form
            for i = 0:4
                Q = Matrix{T}(I, n, n)
                H = triu(randn(T, n, n))
                H[1+i:n-i, 1+i:n-i] = normal_hessenberg_matrix(T, i+1:n-i)
                H′ = copy(H)

                # Test that the procedure has converged
                @test local_schurfact!(H′, 1 + i, n - i, Q)

                for j = 1+i:n-i-1
                    t = H′[j, j] + H′[j+1, j+1]
                    d = H′[j, j] * H′[j+1, j+1] - H′[j+1, j] * H′[j, j+1]

                    # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
                    @test is_offdiagonal_small(H′, j) || t^2 < 4d
                end

                # Test that the elements below the subdiagonal are 0
                @test is_hessenberg(H′)

                # Test that the partial Schur decomposition relation holds
                @test norm(H * Q - Q * H′) < 1000eps(T)

                # Test that the eigenvalues of H are the same before and after transformation
                @test sort!(eigvals(H), by = realimag) ≈ sort!(eigvals(H′), by = realimag)
            end
        end



        # COMPLEX ARITHMETIC

        let
            n = 10
            # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
            for i = 0:4
                Q = Matrix{Complex{T}}(I, n, n)
                H = triu(randn(Complex{T}, n, n))
                H[i+1:n-i, i+1:n-i] =
                    normal_hessenberg_matrix(Complex{T}, (i+1:n-i) .* (1 + im))
                H′ = copy(H)

                # Test that the procedure has converged
                @test local_schurfact!(H′, 1 + i, n - i, Q)

                # Test if subdiagonal is small. 
                for j = 1+i:n-i-1
                    @test iszero(H′[j+1, j])
                end

                # Test that the elements below the subdiagonal are 0
                @test is_hessenberg(H′)

                # Test that the partial Schur decomposition relation holds
                @test norm(H * Q - Q * H′) < 1000eps(T)

                # Test that the eigenvalues of H are the same before and after transformation
                @test sort!(eigvals(H), by = realimag) ≈ sort!(eigvals(H′), by = realimag)
            end
        end
    end
end

@testset "Schur with nearly repeated eigenvalues" begin
    # Matrix with nearly repeated eigenpair could converge very slowly or
    # stagnate completely when there's a tiny perturbation in the computed
    # shift.
    mat(ε) = [
        2 0 0
        5ε 1-ε 2ε
        0 3ε 1+ε
    ]
    for T in (Float64, BigFloat)
        @test local_schurfact!(mat(eps(T)))
    end
end

@testset "Convergence issue encountered in the wild" begin
    # This 4x4 matrix with almost identical eigenvalues previously caused tens of thousands
    # of iterations of the QR algorithm to converge, likely due to unstable computation of
    # shifts and (H - μ₁I)(H - μ₂I)e₁ column.
    H = [
        -9.000000046596169 9.363971416904122e-6 0.6216202324428521 0.783119615978767
        -3.1249216068055166e-10 -9.000000125049475 -0.005030734831215954 0.026538692060151765
        0.0 2.5838932886290116e-12 -8.999999884550379 -4.118678562647915e-7
        0.0 0.0 5.499735555858365e-9 -8.99999994380397
    ]
    @test local_schurfact!(H, 1, 4)
end
