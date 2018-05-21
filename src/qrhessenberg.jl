module QRHessenberg

using Base.LinAlg: givensAlgorithm
using Base.@propagate_inbounds

import Base: getindex

export MyHessenberg, qr_hessenberg!

struct MyGivens{Tv,Ti}
  c::Tv
  s::Tv
  i::Ti
end

struct ListOfRotations{Tv}
  rotations::Vector{Tuple{Tv,Tv}}
end

struct MyHessenberg{T}
    H::T
end

@propagate_inbounds function getindex(Q::ListOfRotations, i) 
  c, s = Q.rotations[i]
  MyGivens(c, s, i)
end

function mul!(A::AbstractMatrix, Q::ListOfRotations)
  for i = 1 : length(Q.rotations)
    mul!(A, Q[i])
  end
  A
end

function mul!(G::MyGivens, H::MyHessenberg)
    dim = size(H.H, 1)
    @inbounds for j in G.i:dim
        h_min = G.c.*H.H[G.i,j] + G.s*H.H[G.i+1,j]
        h_max = -conj(G.s)*H.H[G.i,j] + G.c*H.H[G.i+1,j]
        H.H[G.i,j] = h_min
        H.H[G.i+1,j] = h_max
    end
end

function mul!(A::AbstractMatrix, G::MyGivens)
    dim = size(A, 1)
    @inbounds for j in 1:dim
        a_min = G.c*A[j,G.i] + G.s*A[j,G.i+1]
        a_max = -conj(G.s)*A[j,G.i] + G.c*A[j,G.i+1]
        A[j,G.i] = a_min
        A[j,G.i+1] = a_max
    end
end

function qr_hessenberg!(H::MyHessenberg)
    dim = size(H.H, 1)
    Q = MyHessenberg(eye(dim))
    list = ListOfRotations(Vector{Tuple{Float64,Float64}}(dim-1))
    for i in 1:dim-1
        c, s = givensAlgorithm(H.H[i,i],H.H[i+1,i])
        mul!(MyGivens(c, s, i), H)
        list.rotations[i] = (c,s)
    end
    mul!(Q.H, list)
    (MyHessenberg(Q.H'),H,list)
end

end