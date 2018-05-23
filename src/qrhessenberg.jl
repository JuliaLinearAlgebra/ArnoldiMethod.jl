using Base.LinAlg: givensAlgorithm
using Base.@propagate_inbounds

import Base: getindex

export Hessenberg, qr!, mul!, ListOfRotations

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
    ListOfRotations{Tc,Ts}(Vector{Tuple{Tc,Ts}}(total))
end

struct Hessenberg{T}
    H::T
end

@propagate_inbounds function getindex(Q::ListOfRotations, i) 
    c, s = Q.rotations[i]
    Givens(c, s, i)
end

function mul!(A::AbstractMatrix, Q::ListOfRotations)
    for i = 1 : size(A, 2) - 1
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

"""
Applies the Givens rotation to Hessenberg matrix H from the left (in-place).
"""
function mul!(G::Givens, H::Hessenberg)
    @inbounds for j in G.i:size(H.H, 2)
        h_min = G.c .* H.H[G.i,j] + G.s * H.H[G.i + 1,j]
        h_max = -conj(G.s) * H.H[G.i,j] + G.c * H.H[G.i + 1,j]
        H.H[G.i,j] = h_min
        H.H[G.i + 1,j] = h_max
    end
    H
end

"""
Applies the transpose of the Givens rotation to A from the right (in-place).
"""
function mul!(A::AbstractMatrix, G::Givens)
    dim = size(A, 2)
    @inbounds for j in 1:dim
        a_min = G.c * A[j,G.i] + conj(G.s) * A[j,G.i + 1]
        a_max = -G.s * A[j,G.i] + G.c * A[j,G.i + 1]
        A[j,G.i] = a_min
        A[j,G.i + 1] = a_max
    end
    A
end

"""
Apply Given's rotations to H so that it becomes upper triangular (in-place).

Stores the list of Givens rotations in L (in-place).
"""
function qr!(H::Hessenberg, L::ListOfRotations)
    dim = size(H.H, 1)
    
    for i in 1:dim - 1
        # Find new Givens coefficients
        c, s = givensAlgorithm(H.H[i,i], H.H[i + 1,i])

        # Apply the rotation
        mul!(Givens(c, s, i), H)

        # Store the rotation
        L.rotations[i] = (c, s)
    end

    L
end
