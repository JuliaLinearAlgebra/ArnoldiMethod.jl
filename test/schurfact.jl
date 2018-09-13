# Test if we can transform part of a Hessenberg matrix to Schur form.
# Here we look at some edge cases.

using Test, LinearAlgebra
using ArnoldiMethod: eigenvalues, local_schurfact!, is_offdiagonal_small,
                     NotWanted

include("utils.jl")

@testset "Schur factorization" begin
    let
        # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
        H = [1.0 2.0; 3.0 4.0]
        H′ = copy(H)
        Q = Matrix{Float64}(I, 2, 2)
        
        @test local_schurfact!(H′, 1, 2, Q, eps(), 2)
        @test norm(H * Q - Q * H′) < 10eps()
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H′), by=realimag)
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H), by=realimag)
        @test iszero(H′[2,1])
    end

    let
        # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
        H = [1.0 2.0; 0.0 4.0]
        H′ = copy(H)
        Q = Matrix{Float64}(I, 2, 2)
        
        @test local_schurfact!(H′, 1, 2, Q, eps(), 2)
        @test norm(H * Q - Q * H′) < 10eps()
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H′), by=realimag)
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H), by=realimag)
        @test iszero(H′[2,1])
    end

    let
        # 2-by-2 matrix with conjugate eigenvalues
        H = [1.0 4.0; -5.0 3.0]
        H′ = copy(H)
        Q = Matrix{Float64}(I, 2, 2)
        
        @test local_schurfact!(H′, 1, 2, Q, eps(), 2)
        @test norm(H * Q - Q * H′) < 10eps()
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H′), by=realimag)
        @test sort!(eigenvalues(H′), by=realimag) ≈ sort!(eigvals(H), by=realimag)
    end
    
    # Larger real matrix.

    let
        n = 10
        # Transforming a 1+i by n-i block of the matrix H into upper triangular form
        for i = 0 : 4
            Q = Matrix{Float64}(I, n, n)
            H = triu(randn(Float64, n, n))
            H[1+i:n-i,1+i:n-i] = normal_hessenberg_matrix(Float64, i+1:n-i)
            H′ = copy(H)

            # Test that the procedure has converged
            @test local_schurfact!(H′, 1+i, n-i, Q)

            for j = 1+i:n-i-1
                t = H′[j,j] + H′[j+1,j+1]
                d = H′[j,j] * H′[j+1,j+1] - H′[j+1,j] * H′[j,j+1]

                # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
                @test is_offdiagonal_small(H′, j) || t^2 < 4d
            end

            # Test that the elements below the subdiagonal are 0
            @test is_hessenberg(H′)

            # Test that the partial Schur decomposition relation holds
            @test norm(H * Q - Q * H′) < 1000eps()
            
            # Test that the eigenvalues of H are the same before and after transformation
            @test sort!(eigvals(H), by=realimag) ≈ sort!(eigvals(H′), by=realimag)
        end

    end


    # COMPLEX ARITHMETIC

    let
        n = 10
        # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
        for i = 0 : 4
            Q = Matrix{ComplexF64}(I, n, n)
            H = triu(randn(ComplexF64, n, n))
            H[i+1:n-i,i+1:n-i] = normal_hessenberg_matrix(ComplexF64, (i+1:n-i) .* (1 + im))
            H′ = copy(H)

            # Test that the procedure has converged
            @test local_schurfact!(H′, 1+i, n-i, Q)

            # Test if subdiagonal is small. 
            for j = 1+i:n-i-1
                @test iszero(H′[j+1,j])
            end

            # Test that the elements below the subdiagonal are 0
            @test is_hessenberg(H′)

            # Test that the partial Schur decomposition relation holds
            @test norm(H * Q - Q * H′) < 1000eps()
            
            # Test that the eigenvalues of H are the same before and after transformation
            @test sort!(eigvals(H), by=realimag) ≈ sort!(eigvals(H′), by=realimag)
        end
    end
end

@testset "Schur with nearly repeated eigenvalues" begin
    # Matrix with nearly repeated eigenpair could converge very slowly or
    # stagnate completely when there's a tiny perturbation in the computed
    # shift.
    mat(ε) = [2   0    0  ;
              5ε  1-ε  2ε ;
              0   3ε   1+ε]

    @test local_schurfact!(mat(eps()))
end