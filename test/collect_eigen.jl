using Test

using ArnoldiMethod: collect_eigen!, copy_eigenvalues!
using LinearAlgebra

@testset "Eigenvector upper triangular matrix" begin
    n = 20

    # Exactly upper triangular matrix R
    for T in (Float64, ComplexF64)

        # Upper triangular matrix
        R = triu(rand(T, n, n))

        # Compute exact eigenvectors according to LAPACK
        λs, xs = eigen(R)

        # Allocate a complex-valued eigenvector
        x = zeros(complex(T), n)

        for i = 1:n
            # Compute the `i`th eigenvector
            fill!(x, zero(T))
            collect_eigen!(x, R, i)

            # Test whether it corresponds to the LAPACK result
            @test norm(x) ≈ 1
            @test abs.(x) ≈ abs.(xs[:, i])
        end
    end
end

rot(θ) = [cos(θ) sin(θ); -sin(θ) cos(θ)]

@testset "Eigenvectors quasi-upper triangular matrix" begin
    # Real arithmetic with conjugate eigvals -- quasi-upper triangular R
    n = 20
    R = triu(rand(n, n))
    R[1:2,1:2] .= rot(1.0) + I
    R[10:11,10:11] .= rot(1.2) + 2I
    
    # Compute exact eigenvectors according to LAPACK
    λs, xs = eigen(R)

    # Allocate a complex-valued eigenvector
    x = zeros(ComplexF64, n)

    for i = 1:n
        fill!(x, 0.0)
        collect_eigen!(x, R, i)

        @test norm(x) ≈ 1
        @test abs.(x) ≈ abs.(xs[:, i])
    end
end

@testset "Extract partial" begin
    n = 20
    R = triu(rand(n, n))
    R[1:2,1:2] .= rot(1.0) + I
    for range = (1:3, 1:4)
        λs = eigvals(R[range, range])
        θs = copy_eigenvalues!(rand(ComplexF64, length(range)), R, range)
        @test sort!(λs, by = reim) ≈ sort!(θs, by = reim)
    end
end