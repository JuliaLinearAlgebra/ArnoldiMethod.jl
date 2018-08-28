# Core part of implicit restart is to shift away unwanted eigenvalues from the
# non-square Hessenberg matrix. Here we test the single shift on a real Hessenberg
# matrix with a real eigenvalue, and a random complex Hessenberg matrix.
# Once it is shifted away, the other eigenvalues should be retained (in exact arithmetic)

using Test, LinearAlgebra
using ArnoldiMethod: exact_single_shift!

include("utils.jl")

# Generate a real normal hessenberg matrix with eigenvalues in [0.5, 1.0]
function normal_hess_real_eigvals(T::Type{<:Real}, n::Int)
    Htop = normal_hessenberg_matrix(T, range(0.5, stop=1.0, length=n))
    H = [Htop; zeros(T, n)']
    H[end,end] = one(T)
    λs = sort!(eigvals(Htop), by = realimag)
    μ = popfirst!(λs)
    return H, λs, μ
end

# Generate a complex normal hessenberg matrix with eigenvalues on the line y = x
# in the complex plane
function normal_hess_imag_eigvals(T::Type{<:Complex}, n::Int)
    vals = range(0.5, stop=1.0, length=n) .+ im .* collect(range(0.5, stop=1.0, length=n))
    Htop = normal_hessenberg_matrix(T, vals)
    H = [Htop; zeros(T, n)']
    H[end,end] = one(T)
    λs = sort!(eigvals(Htop), by=realimag)
    μ = popfirst!(λs)
    return H, λs, μ
end

@testset "Single Shifted QR" begin
    n = 20

    # Real arithmetic
    for i = 1 : 50
        H, λs, μ = normal_hess_real_eigvals(Float64, n)
        Q = Matrix{Float64}(I, n, n)
        exact_single_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by=realimag)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end

    # Complex arithmethic
    for i = 1 : 50
        H, λs, μ = normal_hess_imag_eigvals(ComplexF64, n)
        Q = Matrix{ComplexF64}(I, n+1, n+1)
        H_copy = copy(H)
        exact_single_shift!(H, 1, n, μ, Q)

        # Update Q with the last rotation
        Q[1:n+1, n] .= 0
        Q[n+1,n] = 1

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(view(H, 1:n-1, 1:n-1)), by=realimag)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)

        # Test whether relation " H_prev * Q = Q * H_next " holds
        @test norm(H_copy * Q[1:n,1:n-1] - Q[1:n+1,1:n] * H[1:n,1:n-1]) < 1e-6
        @test norm(Q[1:n,1:n-1]' * H_copy[1:n,1:n] * Q[1:n,1:n-1] - H[1:n-1,1:n-1]) < 1e-6
    end
end
