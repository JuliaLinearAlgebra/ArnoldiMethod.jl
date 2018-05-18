module qrhessenberg

using Base.LinAlg.givensAlgorithm

struct MyGivens{T,I}
    c::T
    s::T
    i::I
end

struct MyHessenberg{T}
    H::T
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
    for i in 2:dim
        c, s = givensAlgorithm(H.H[i-1,i-1],H.H[i,i-1])
        rotation = MyGivens(c, s, i-1)
        mul!(rotation, H)
        mul!(Q.H, rotation)
    end
    (MyHessenberg(Q.H'),H)
end

end