# Core part of implicit restart is to shift away unwanted eigenvalues from the
# non-square Hessenberg matrix. Here we test the double shift on a real 
# Hessenberg matrix with a conjugate eigenpair; once it is shifted away, the
# other eigenvalues should be retained (in exact arithmetic)

using Test, LinearAlgebra
using IRAM: exact_double_shift!

include("utils.jl")

# Generate a real normal hessenberg matrix with eigenvalues in [0.5, 1.0]
function normal_hess_conjugate_eigvals(T::Type{<:Real}, n::Int)
    Htop = normal_hessenberg_matrix(T, range(0.5, stop=1.0, length=n))
    H = [Htop; zeros(T, n)']
    H[end,end] = one(T)
    λs = sort!(eigvals(Htop), by = realimag)
    μ = popfirst!(λs)
    return H, λs, μ
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
        @test λs ≈ sort!(eigvals(view(H, 1:n-2, 1:n-2)), by=realimag)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end
end
