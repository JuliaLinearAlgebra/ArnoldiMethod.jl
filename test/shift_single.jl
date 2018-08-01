# Core part of implicit restart is to shift away unwanted eigenvalues from the
# non-square Hessenberg matrix. Here we test the single shift on a real Hessenberg
# matrix with a real eigenvalue, and a random complex Hessenberg matrix.
# Once it is shifted away, the other eigenvalues should be retained (in exact arithmetic)

using Test, LinearAlgebra
using IRAM: exact_single_shift!

# Generates a real Hessenberg matrix with one real eigenvalue
# which can be used for testing the single shift in real arithmetic
function generate_real_H_with_real_eigs(n, T::Type = Float64)
    while true
        H = triu(rand(T, n + 1, n), -1)
        λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = x -> (real(x), imag(x)))

        for i = 1 : n
            μ = λs[i]
            if imag(μ) == 0
                deleteat!(λs, i)
                return H, λs, real(μ)
            end
        end
    end
end

# Similarly generates a complex Hessenberg matrix, with any eigenvalue
# that can be used to test a single shift in complex arithmetic
function generate_complex_H(n, T::Type = ComplexF64)
    H = triu(rand(T, n + 1, n), -1)
    λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = x -> (real(x), imag(x)))
    μ = λs[1]
    deleteat!(λs, 1)
    return H, λs, μ
end

@testset "Single Shifted QR" begin
    n = 20

    is_hessenberg(H) = norm(tril(H, -2)) == 0

    # Real arithmetic
    for i = 1 : 50
        H, λs, μ = generate_real_H_with_real_eigs(n, Float64)
        Q = Matrix{Float64}(I, n, n)
        # H_copy = copy(H)

        exact_single_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by = x -> (real(x), imag(x)))

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end

    # Complex arithmethic
    for i = 1 : 50
        H, λs, μ = generate_complex_H(n, ComplexF64)
        Q = Matrix{ComplexF64}(I, n, n)

        exact_single_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by = x -> (real(x), imag(x)))

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end
end
