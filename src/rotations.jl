using LinearAlgebra: givensAlgorithm
using Base: @propagate_inbounds

import Base: getindex, setindex!, size
import LinearAlgebra: rmul!, lmul!

export Hessenberg, qr!, ListOfRotations

struct Givens{Tc,Ts,Ti}
    c::Tc
    s::Ts
    i::Ti
end

struct ListOfRotations{Tc,Ts}
    rotations::Vector{Tuple{Tc,Ts}}
end

function ListOfRotations(Ts::Type, total::Int)
    Tc = real(Ts)
    ListOfRotations{Tc,Ts}(Vector{Tuple{Tc,Ts}}(undef, total))
end

struct Hessenberg{T,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    H::V
end

Hessenberg(H::AbstractMatrix) = Hessenberg{eltype(H),typeof(H)}(H)
@propagate_inbounds getindex(H::Hessenberg, I...) = getindex(H.H, I...)
@propagate_inbounds setindex!(H::Hessenberg, I...) = setindex!(H.H, I...)
size(H::Hessenberg) = size(H.H)
size(H::Hessenberg, d) = size(H.H, d)

struct UpperTriangularMatrix{T,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    R::V
end

UpperTriangularMatrix(R::AbstractMatrix) = UpperTriangularMatrix{eltype(R),typeof(R)}(R)
@propagate_inbounds getindex(R::UpperTriangularMatrix, I...) = getindex(R.R, I...)
@propagate_inbounds setindex!(R::UpperTriangularMatrix, I...) = setindex!(R.R, I...)
size(R::UpperTriangularMatrix) = size(R.R)
size(R::UpperTriangularMatrix, d) = size(R.R, d)

@propagate_inbounds function getindex(Q::ListOfRotations, i) 
    c, s = Q.rotations[i]
    Givens(c, s, i)
end

function rmul!(A::AbstractMatrix, Q::ListOfRotations)
    @inbounds for i = 1 : size(A, 2) - 1
        rmul!(A, Q[i])
    end
    A
end

function lmul!(Q::ListOfRotations, H::Hessenberg)
    @inbounds for i = 1:length(Q.rotations)
        lmul!(Q[i], H)
    end
    H
end

"""
Applies the Givens rotation to Hessenberg matrix H from the left (in-place).
"""
function lmul!(G::Givens, H::Hessenberg)
    @inbounds for j in G.i:size(H, 2)
        h_min = G.c .* H[G.i,j] + G.s * H[G.i + 1,j]
        h_max = -conj(G.s) * H[G.i,j] + G.c * H[G.i + 1,j]
        H[G.i,j] = h_min
        H[G.i + 1,j] = h_max
    end
    H
end

function lmul!(G::Givens, H::AbstractMatrix)
    @inbounds for j in 1:size(H, 2)
        h_min = G.c .* H[G.i,j] + G.s * H[G.i + 1,j]
        h_max = -conj(G.s) * H[G.i,j] + G.c * H[G.i + 1,j]
        H[G.i,j] = h_min
        H[G.i + 1,j] = h_max
    end
    H
end

"""
Applies the transpose of the Givens rotation to A from the right (in-place).
"""
function rmul!(A::AbstractMatrix, G::Givens)
    dim = size(A, 1)
    @inbounds for j in 1:dim
        a_min = G.c * A[j,G.i] + conj(G.s) * A[j,G.i + 1]
        a_max = -G.s * A[j,G.i] + G.c * A[j,G.i + 1]
        A[j,G.i] = a_min
        A[j,G.i + 1] = a_max
    end
    A
end

"""
Applies the transpose of the Givens rotation to R from the right (in-place).
"""
function rmul!(R::UpperTriangularMatrix, G::Givens)
    @inbounds for j in 1:G.i + 1
        a_min = G.c * R[j,G.i] + conj(G.s) * R[j,G.i + 1]
        a_max = -G.s * R[j,G.i] + G.c * R[j,G.i + 1]
        R[j,G.i] = a_min
        R[j,G.i + 1] = a_max
    end
    R
end

"""
Apply Given's rotations to H so that it becomes upper triangular (in-place).

Stores the list of Givens rotations in L (in-place).
"""
function qr!(H::Hessenberg, L::ListOfRotations)
    dim = size(H, 1)
    
    @inbounds for i in 1:dim - 1
        # Find new Givens coefficients
        c, s = givensAlgorithm(H[i,i], H[i + 1,i])

        # Apply the rotation
        lmul!(Givens(c, s, i), H)

        # Store the rotation
        L.rotations[i] = (c, s)
    end

    L
end
