module ArnoldiMethod

using LinearAlgebra
using StaticArrays

using Base: RefValue, OneTo

export partialschur, partialschur!, partialeigen, ArnoldiWorkspace

"""
    ArnoldiWorkspace(n, k) → ArnoldiWorkspace
    ArnoldiWorkspace(v1, k) → ArnoldiWorkspace
    ArnoldiWorkspace(V, H; V_tmp, Q) → ArnoldiWorkspace

Holds the large arrays for the Arnoldi relation: Vₖ₊₁ and Hₖ are matrices that
satisfy A * Vₖ = Vₖ₊₁ * Hₖ, where Vₖ₊₁ is orthonormal of size n × (k+1) and Hₖ upper 
Hessenberg of size (k+1) × k. The matrices V_tmp and Q are used for restarts, and
have similar size as Vₖ₊₁ and Hₖ (but Q is k × k, not k+1 × k).

## Examples

```julia
# allocates workspace for 20-dimensional Krylov subspace
arnoldi = ArnoldiWorkspace(100, 20)

# allocate workspace for 20-dimensional Krylov subspace, with initial vector ones(100) copied into
# the first column of V
arnoldi = ArnoldiWorkspace(ones(100), 20)

# manually allocate workspace with V, H
V = Matrix{Float64}(undef, 100, 21)
H = Matrix{Float64}(undef, 21, 20)
arnoldi = ArnoldiWorkspace(V, H)

# manually allocate all arrays, including temporaries
V, tmp = Matrix{Float64}(undef, 100, 21), Matrix{Float64}(undef, 100, 21)
H, Q = Matrix{Float64}(undef, 21, 20), Matrix{Float64}(undef, 20, 20)
arnoldi = ArnoldiWorkspace(V, H, V_tmp = tmp, Q = Q)
```
"""
struct ArnoldiWorkspace{
    T,
    TV<:AbstractMatrix{T},
    TH<:AbstractMatrix{T},
    TVtmp<:AbstractMatrix{T},
    TQ<:AbstractMatrix{T},
}
    # Orthonormal basis of the Krylov subspace.
    V::TV

    # The Hessenberg matrix of the Arnoldi relation.
    H::TH

    # Temporary matrix similar to V, used to restart.
    V_tmp::TVtmp

    # Unitary matrix size of (square) H to do a change of basis.
    Q::TQ

    function ArnoldiWorkspace(::Type{T}, matrix_order::Int, krylov_dimension::Int) where {T}
        # Without an initial vector.
        krylov_dimension <= matrix_order ||
            throw(ArgumentError("Krylov dimension should be less than matrix order."))
        V = Matrix{T}(undef, matrix_order, krylov_dimension + 1)
        V_tmp = similar(V)
        H = zeros(T, krylov_dimension + 1, krylov_dimension)
        Q = similar(H, krylov_dimension, krylov_dimension)
        return new{T,typeof(V),typeof(H),typeof(V_tmp),typeof(Q)}(V, H, V_tmp, Q)
    end

    function ArnoldiWorkspace(v1::AbstractVector{T}, krylov_dimension::Int) where {T}
        # From an initial vector v1.
        V = similar(v1, length(v1), krylov_dimension + 1)
        V_tmp = similar(V)
        H = similar(V, krylov_dimension + 1, krylov_dimension)
        fill!(H, zero(eltype(H)))
        Q = similar(H, krylov_dimension, krylov_dimension)
        return new{T,typeof(V),typeof(H),typeof(V_tmp),typeof(Q)}(V, H, V_tmp, Q)
    end

    function ArnoldiWorkspace(
        V::AbstractMatrix{T},
        H::AbstractMatrix{T};
        V_tmp::AbstractMatrix{T} = similar(V),
        Q::AbstractMatrix{T} = similar(H, size(H, 2), size(H, 2)),
    ) where {T}
        size(V, 2) == size(H, 1) ||
            throw(ArgumentError("V should have the same number of columns as H has rows."))
        size(H, 1) == size(H, 2) + 1 ||
            throw(ArgumentError("H should have one more row than it has columns."))
        return new{T,typeof(V),typeof(H),typeof(V_tmp),typeof(Q)}(V, H, V_tmp, Q)
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
