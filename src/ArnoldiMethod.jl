module ArnoldiMethod

using LinearAlgebra
using StaticArrays

using Base: RefValue, OneTo

export partialschur, LM, SR, LR, SI, LI, partialeigen
export History, GeneralHistory, DetailedHistory

"""
    Arnoldi(n, k) → Arnoldi

Pre-allocated Arnoldi relation of the Vₖ₊₁ and Hₖ matrices that satisfy
A * Vₖ = Vₖ₊₁ * Hₖ, where Vₖ₊₁ is orthonormal of size n × (k+1) and Hₖ upper
Hessenberg of size (k+1) × k. The constructor will just allocate sufficient
space, but will *not* initialize the first vector of `v₁`. For the latter see
`reinitialize!`.
"""
struct Arnoldi{T,TV<:StridedMatrix{T},TH<:StridedMatrix{T}}
    V::TV
    H::TH

    function Arnoldi{T}(matrix_order::Int, krylov_dimension::Int) where {T}
        krylov_dimension <= matrix_order || throw(ArgumentError("Krylov dimension should be less than matrix order."))
        V = Matrix{T}(undef, matrix_order, krylov_dimension + 1)
        H = zeros(T, krylov_dimension + 1, krylov_dimension)
        return new{T,typeof(V),typeof(H)}(V, H)
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
matrices with complex eigenvalues. Note that the eigenvalues are always a
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
include("partition.jl")
include("schurfact.jl")
include("schursort.jl")
include("restore_hessenberg.jl")
include("expansion.jl")
include("run.jl")
include("eigvals.jl")
include("eigenvector_uppertriangular.jl")
include("show.jl")


end
