# Core part of implicit restart is to shift away unwanted eigenvalues from the
# non-square Hessenberg matrix. Here we test the double shift on a real 
# Hessenberg matrix with a conjugate eigenpair; once it is shifted away, the
# other eigenvalues should be retained (in exact arithmetic)

using Test, LinearAlgebra
using IRAM: exact_double_shift!

function generate_real_H_with_imaginary_eigs(n, T::Type = Float64)
    while true
        H = triu(rand(T, n + 1, n), -1)
        λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs)

        for i = 1 : n
            μ = λs[i]
            if imag(μ) != 0
                # Remove conjugate pair
                deleteat!(λs, (i, i + 1))
                return H, λs, μ
            end
        end
    end
end

@testset "Double Shifted QR" begin
    n = 20

    is_hessenberg(H) = norm(tril(H, -2)) == 0

    # Test on a couple random matrices
    for i = 1 : 50
        H, λs, μ = generate_real_H_with_imaginary_eigs(n, Float64)
        Q = Matrix{Float64}(I, n, n)
        exact_double_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-2, 1:n-2)), by = abs)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end
end
