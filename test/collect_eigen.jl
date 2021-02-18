using Test

using ArnoldiMethod: collect_eigen!, copy_eigenvalues!
using LinearAlgebra

@testset "Eigenvector upper triangular matrix" begin
    n = 20

    # Exactly upper triangular matrix R
    for T in (Float64, ComplexF64, BigFloat, Complex{BigFloat})

        # Upper triangular matrix
        R = triu(rand(T, n, n))

        # Compute exact eigenvectors according to LAPACK
        if VERSION >= v"1.2.0-DEV.275"
            λs, xs = eigen(R, sortby=nothing)
        else
            λs, xs = eigen(R)
        end

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
    for T in (Float64, BigFloat)
        n = 20
        R = triu(rand(T, n, n))
        R[1:2,1:2] .= rot(one(T)) + I
        R[10:11,10:11] .= rot(T(6//5)) + 2I
        
        # Compute exact eigenvectors according to LAPACK
        if VERSION >= v"1.2.0-DEV.275"
            λs, xs = eigen(R, sortby=nothing)
        else
            λs, xs = eigen(R)
        end

        # Allocate a complex-valued eigenvector
        x = zeros(Complex{T}, n)

        for i = 1:n
            fill!(x, 0)
            collect_eigen!(x, R, i)

            @test norm(x) ≈ 1
            @test abs.(x) ≈ abs.(xs[:, i])
        end
    end
end

@testset "Extract partial" begin
    n = 20
    for T in (Float64, BigFloat)
        R = triu(rand(T,n, n))
        R[1:2,1:2] .= rot(1.0) + I
        for range = (1:3, 1:4)
            λs = eigvals(R[range, range])
            θs = copy_eigenvalues!(rand(Complex{T}, length(range)), R, range)
            @test sort!(λs, by = reim) ≈ sort!(θs, by = reim)
        end
    end
end
