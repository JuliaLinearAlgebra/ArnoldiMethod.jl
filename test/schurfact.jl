using Test

using IRAM: eigvalues, local_schurfact!

@testset "Schur factorization" begin

    # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
    H = [1.0 2.0; 3.0 4.0]
    H_copy = copy(H)
    Q = Matrix{Float64}(I, 2, 2)
    local_schurfact!(H, Q, 1, 2)

    @test norm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
    H = [1.0 2.0; 0.0 4.0]
    H_copy = copy(H)
    Q = Matrix{Float64}(I, 2, 2)
    local_schurfact!(H, Q, 1, 2)

    @test norm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with conjugate eigenvalues
    H = [1.0 4.0; -5.0 3.0]
    H_copy = copy(H)
    Q = Matrix{Float64}(I, 2, 2)
    local_schurfact!(H, Q, 1, 2)

    @test norm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8
    
    # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    for i = 0 : 4
        H = triu(rand(10,10), -1)

        # Add zeros to the subdiagonals assuming convergence
        if i != 0 
            # The previous block has converged, hence H[i+1,i] = 0
            H[i+1,i] = 0

            # Current block has converged, hence H[11-i,10-i] = 0
            H[11-i,10-i] = 0
        end
        λs = sort!(eigvals(H), by=abs, rev=true)
        H_copy = copy(H)
        Q = Matrix{Float64}(I, 10, 10)

        # Test that the procedure has converged
        @test local_schurfact!(H, Q, 1+i, 10-i)

        for j = 1 : 9 - 2*i
            t = H[i+j,i+j] + H[i+j+1,i+j+1]
            d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

            # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
            @test H[i+j+1,i+j] < 1e-8 || t*t/4 - d < 0
        end

        # Test that the elements below the subdiagonal are 0
        for j = 1:10, i = j+2:10
            @test H[i,j] < 1e-8
        end

        # Test that the partial Schur decomposition relation holds
        @test norm(Q*H*Q' - H_copy) < 1e-8
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
    end


    # COMPLEX ARITHMETIC

    # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    for i = 0 : 4
        H = triu(rand(ComplexF64, 10,10), -1)

        # Add zeros to the subdiagonals assuming convergence
        if i != 0 
            # The previous block has converged, hence H[i+1,i] = 0
            H[i+1,i] = zero(ComplexF64)

            # Current block has converged, hence H[11-i,10-i] = 0
            H[11-i,10-i] = zero(ComplexF64)
        end

        λs = sort!(eigvals(H), by=abs, rev=true)
        H_copy = copy(H)
        Q = Matrix{ComplexF64}(I, 10, 10)

        # Test that the procedure has converged
        local_schurfact!(H, Q, 1+i, 10-i)

        for j = 1 : 9 - 2*i  
            # Test if subdiagonal is small. 
            @test abs(H[i+j+1,i+j]) < 1e-8
        end

        # Test that the elements below the subdiagonal are 0
        for j = 1:10
            for i = j+2:10
                @test abs(H[i,j]) < 1e-8
            end
        end

        # Test that the partial Schur decomposition relation holds
        @test norm(Q*H*Q' - H_copy) < 1e-8
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
    end
end
