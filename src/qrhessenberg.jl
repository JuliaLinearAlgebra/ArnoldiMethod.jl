using Base.LinAlg: givensAlgorithm
using Base.@propagate_inbounds

import Base: getindex

export Hessenberg, qr!, mul!

struct Givens{Tv,Ti}
    c::Tv
    s::Tv
    i::Ti
end

struct ListOfRotations{Tv}
    rotations::Vector{Tuple{Tv,Tv}}
end

ListOfRotations(T::Type, total::Int) = ListOfRotations{T}(Vector{Tuple{T,T}}(total))

struct Hessenberg{T}
    H::T
end

@propagate_inbounds function getindex(Q::ListOfRotations, i) 
    c, s = Q.rotations[i]
    Givens(c, s, i)
end

function mul!(A::AbstractMatrix, Q::ListOfRotations)
    for i = 1:length(Q.rotations)
        mul!(A, Q[i])
    end
    A
end

function mul!(Q::ListOfRotations, H::Hessenberg)
    for i = 1:length(Q.rotations)
        mul!(Q[i], H)
    end
    H
end

function mul!(G::Givens, H::Hessenberg)
    @inbounds for j in G.i:size(H.H, 2)
        h_min = G.c .* H.H[G.i,j] + G.s * H.H[G.i + 1,j]
        h_max = -conj(G.s) * H.H[G.i,j] + G.c * H.H[G.i + 1,j]
        H.H[G.i,j] = h_min
        H.H[G.i + 1,j] = h_max
    end
end

function mul!(A::AbstractMatrix, G::Givens)
    dim = size(A, 1)
    @inbounds for j in 1:dim
        a_min = G.c * A[j,G.i] + G.s * A[j,G.i + 1]
        a_max = -conj(G.s) * A[j,G.i] + G.c * A[j,G.i + 1]
        A[j,G.i] = a_min
        A[j,G.i + 1] = a_max
    end
end

"""
    qr!(H::Hessenberg) -> ListOfRotations

Apply Given's rotations to H so that it becomes upper triangular (in-place).

Returns a list of Givens rotations.
"""
function qr!(H::Hessenberg)
    dim = size(H.H, 1)
    
    list = ListOfRotations(eltype(H.H), dim - 1)

    for i in 1:dim - 1
        # Find new Givens coefficients
        c, s = givensAlgorithm(H.H[i,i], H.H[i + 1,i])

        # Apply the rotation
        mul!(Givens(c, s, i), H)

        # Store the rotation
        list.rotations[i] = (c, s)
    end

    list
end

