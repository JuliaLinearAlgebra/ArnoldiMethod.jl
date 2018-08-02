# Core part of implicit restart is to shift away unwanted eigenvalues from the
# non-square Hessenberg matrix. Here we test the double shift on a real 
# Hessenberg matrix with a conjugate eigenpair; once it is shifted away, the
# other eigenvalues should be retained (in exact arithmetic)

using Test, LinearAlgebra
using IRAM: exact_double_shift!

include("utils.jl")

function normal_hess_conjugate_eigvals(T::Type{<:Real}, n::Int)
    vals = Vector{complex(T)}(range(0.5, stop=1.0, length=n))
    
    # Add two conjugate eigenpairs
    vals[1] = vals[1] + im
    vals[2] = conj(vals[1])
    vals[4] = vals[4] + im
    vals[5] = conj(vals[4])

    Htop = normal_hessenberg_matrix(T, vals)
    H = [Htop; zeros(T, n)']
    H[end,end] = one(T)

    λs = sort!(eigvals(Htop), by = realimag)
    μ = first(λs)
    λs = λs[3:end]
    return H, λs, μ
end

@testset "Double Shifted QR" begin
    n = 20

    # Test on a couple random matrices
    for i = 1 : 50
        H, λs, μ = normal_hess_conjugate_eigvals(Float64, n)
        Q = Matrix{Float64}(I, n+1, n+1)
        H_copy = copy(H)
        exact_double_shift!(H, 1, n, μ, Q)

        # Update Q with the last rotation
        Q[1:n+1, n-1:n] .= 0
        Q[n+1, n-1] = 1

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(H[1:n-2,1:n-2]), by=realimag)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)

        # Test whether relation " H_prev * Q = Q * H_next " holds
        @test norm(H_copy * Q[1:n,1:n-2] - Q[1:n+1,1:n-1] * H[1:n-1,1:n-2]) < 1e-6
        @test norm(Q[1:n,1:n-2]' * H_copy[1:n,1:n] * Q[1:n,1:n-2] - H[1:n-2,1:n-2]) < 1e-6
    end
end
