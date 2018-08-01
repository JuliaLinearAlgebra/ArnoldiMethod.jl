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
        Q = Matrix{Float64}(I, n, n)
        exact_double_shift!(H, 1, n, μ, Q)

        # Test whether exact shifts retain the remaining eigenvalues after the QR step
        @test λs ≈ sort!(eigvals(H[1:n-2,1:n-2]), by=realimag)

        # Test whether the full matrix remains Hessenberg.
        @test is_hessenberg(H)
    end
end
