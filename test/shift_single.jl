using Test

using IRAM: single_shift!

# Generates a real Hessenberg matrix with one real eigenvalue
# which can be used for testing the single shift in real arithmetic
function generate_real_H_with_real_eigs(n, T::Type = Float64)
    while true
        H = triu(rand(T, n + 1, n), -1)
        λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs)

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
    λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs)
    μ = λs[1]
    deleteat!(λs, 1)
    return H, λs, μ
end

@testset "Single Shifted QR" begin
    n = 20

    is_hessenberg(H) = vecnorm(tril(H, -2)) == 0

    # Real arithmetic
    for i = 1 : 50
        H, λs, μ = generate_real_H_with_real_eigs(n, Float64)
        Q = eye(n)
        # H_copy = copy(H)

        single_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by = abs)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)

        # @show(vecnorm(Q[1:n,1:n-1]'*H_copy[1:n,1:n]*Q[1:n,1:n-1] - H[1:n-1,1:n-1]))
    end

    # Complex arithmethic
    for i = 1 : 50
        H, λs, μ = generate_complex_H(n, Complex128)
        Q = eye(Complex128, n)

        single_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by = abs)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end
end
