module ArnoldiMethod

using LinearAlgebra
using StaticArrays

using Base: RefValue, OneTo

export partialschur, LM, SR, LR, SI, LI, partialeigen, ArnoldiWorkspace

"""
ArnoldiWorkspace(n, k) → ArnoldiWorkspace

Holds the large arrays for the Arnoldi relation: Vₖ₊₁ and Hₖ are matrices that
satisfy A * Vₖ = Vₖ₊₁ * Hₖ, where Vₖ₊₁ is orthonormal of size n × (k+1) and Hₖ upper 
Hessenberg of size (k+1) × k.
"""
struct ArnoldiWorkspace{T,TV<:AbstractMatrix{T},TH<:AbstractMatrix{T}}
    V::TV
    V_tmp::TV
    H::TH  # if we support GPU arrays for V, H will still be on the CPU

    function ArnoldiWorkspace(::Type{T}, matrix_order::Int, krylov_dimension::Int) where {T}
        # Without an initial vector.
        krylov_dimension <= matrix_order ||
            throw(ArgumentError("Krylov dimension should be less than matrix order."))
        V = Matrix{T}(undef, matrix_order, krylov_dimension + 1)
        V_tmp = similar(V)
        H = zeros(T, krylov_dimension + 1, krylov_dimension)
        return new{T,typeof(V),typeof(H)}(V, V_tmp, H)
    end

    function ArnoldiWorkspace(v1::AbstractVector{T}, krylov_dimension::Int) where {T}
        # From an initial vector v1.
        V = similar(v1, length(v1), krylov_dimension + 1)
        V_tmp = similar(V)
        H = zeros(T, krylov_dimension + 1, krylov_dimension)
        return new{T,typeof(V),typeof(H)}(V, V_tmp, H)
    end
end

"""
    RitzValues(maxdim) → RitzValues

Convenience wrapper for Ritz values + residual norms and some permutation of 
these values. The Ritz values are computed from the active part of the 
Hessenberg matrix `H[active:maxdim,active:maxdim]`. 

When computing exact shifts in the implicit restart, we need to reorder the Ritz
values in some way. For convenience we simply keep track of a permutation `ord`
of the Ritz values rather than moving the Ritz values themselves around. That
way we don't lose the order of the residual norms.
"""
struct RitzValues{Tv,Tr}
    λs::Vector{Tv}
    rs::Vector{Tr}
    ord::Vector{Int}

    function RitzValues{T}(maxdim::Int) where {T}
        λs = Vector{complex(T)}(undef, maxdim)
        rs = Vector{real(T)}(undef, maxdim)
        ord = Vector{Int}(undef, maxdim)
        return new{complex(T),real(T)}(λs, rs, ord)
    end
end

"""
    PartialSchur(Q, R, eigenvalues)

Holds an orthonormal basis `Q` and a (quasi) upper triangular matrix `R`.

For convenience the eigenvalues that appear on the diagonal of `R` are also 
listed as `eigenvalues`, which is in particular useful in the case of real 
matrices with complex eigenvalues. Note that the eigenvalues are always 
complex, even when the matrix `R` is real.
"""
struct PartialSchur{TQ,TR,Tλ<:Complex}
    "Orthonormal matrix"
    Q::TQ
    "Quasi upper triangular matrix"
    R::TR
    "Complex-valued vector of eigenvalues"
    eigenvalues::Vector{Tλ}
end

include("targets.jl")
include("schurfact.jl")
include("schursort.jl")
include("restore_hessenberg.jl")
include("expansion.jl")
include("run.jl")
include("eigvals.jl")
include("eigenvector_uppertriangular.jl")
include("show.jl")


end
