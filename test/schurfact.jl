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
    

    # # Transforming a 5 by 5 block in different positions of the matrix H into upper triangular form
    # for i = 1 : 6
    #     H = triu(rand(10,10), -1)
    #     λs = sort!(eigvals(H), by=abs, rev=true)
    #     H_copy = copy(H)
    #     Q = eye(10)
    #     schurfact!(H, Q, i, i+4)
    #     λs_transformed = eigvals(H[i:i+4,i:i+4])

    #     for j = 0 : 3
    #         t = H[i+j,i+j] + H[i+j+1,i+j+1]
    #         d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

    #         # Test if sbdiagonal is small. If not, check if conjugate eigenvalues.
    #         @test H[i+j+1,i+j] < 1e-8 || t*t/4 - d < 0
    #         @show H[i+j+1,i+j]
    #         @show (t) * (t) / 4 - (d)
    #         @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
    #     end

    #     display(H)
    #     @test vecnorm(Q[i:i+4,i:i+4]*H[i:i+4,i:i+4]*Q[i:i+4,i:i+4]' - H_copy[i:i+4,i:i+4]) < 1e-8
    #     @test λs ≈ sort!(eigvals(H), by=abs, rev=true)

    # end
    
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
        Q = eye(10)

        # Test that the procedure has converged
        @test schurfact!(H, Q, 1+i, 10-i)

        for j = 1 : 9 - 2*i
            t = H[i+j,i+j] + H[i+j+1,i+j+1]
            d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

            # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
            @test H[i+j+1,i+j] < 1e-8 || t*t/4 - d < 0
            @show H[i+j+1,i+j]
            @show (t) * (t) / 4 - (d)
            @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
        end
        @show 1+i : 10-i
        display(H)
        display(H_copy)

        # Test that the elements below the subdiagonal are 0
        for j = 1:10, i = j+2:10
            @test H[i,j] < 1e-8
        end

        # Test that the partial Schur decomposition relation holds
        @test vecnorm(Q*H*Q' - H_copy) < 1e-8
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
        
        @show sort!(eigvals(H), by=abs, rev=true)
    end


    # COMPLEX ARITHMETIC

    #     # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    # for i = 0 : 4
    #     H = triu(rand(Complex128, 10,10), -1)
    #     λs = sort!(eigvals(H), by=abs, rev=true)
    #     H_copy = copy(H)
    #     Q = eye(Complex128, 10)

    #     # Test that the procedure has converged
    #     schurfact!(H, Q, 1+i, 10-i)

    #     for j = 1 : 9 - 2*i  
    #         # Test if subdiagonal is small. 
    #         @test abs(H[i+j+1,i+j]) < 1e-8
    #         @show H[i+j+1,i+j]
    #         @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
    #     end
    #     @show 1+i : 10-i
    #     display(H)

    #     # Test that the elements below the subdiagonal are 0
    #     for j = 1:10
    #         for i = j+2:10
    #             @test abs(H[i,j]) < 1e-8
    #         end
    #     end

    #     # Test that the partial Schur decomposition relation holds
    #     @test vecnorm(Q*H*Q' - H_copy) < 1e-8
        
    #     # Test that the eigenvalues of H are the same before and after transformation
    #     @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
        
    #     @show sort!(eigvals(H), by=abs, rev=true)
    # end

    
    
end
