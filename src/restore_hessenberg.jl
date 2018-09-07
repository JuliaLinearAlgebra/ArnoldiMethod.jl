"""
    rotate_to_eₖ(y) -> z, τ

Returns z, τ such that (I - 2zz') y = eₖ ‖y‖ τ and |τ| = 1.

Based on Mezzadri, Francesco. "How to generate random matrices from the 
classical compact groups." arXiv preprint math-ph/0609050 (2006).
"""
function rotate_to_eₖ(y, k::Integer = 1)
    @inbounds begin
        τ = y[k] / abs(y[k])
        z = copy(y)
        z[k] += τ * norm(y)
        z ./= norm(z)
        return z, -τ
    end
end

"""
We have the Krylov relation

A[VQ₁ VQ₂] = [VQ₁ VQ₂][R₁₁ R₁₂; + hv[. eₖᵀQ₂]
                       .   R₂₂]

Column VQ₂ has column indices from:to
"""
function restore_hessenberg!(H::AbstractMatrix{T}, from, to, Q) where {T}
    n, m = size(H)
    
    z₁, = rotate_to_eₖ(view(Q, m, from:to), length(from:to))
    W₁ = Matrix{T}(I, m, m)
    W₁[from:to,from:to] .= I - 2 * (z₁ * z₁')

    Q .= Q * W₁
    H[1:m,1:m] .= W₁ * H[1:m,1:m] * W₁

    for i = to:-1:from+2
        z₂, = rotate_to_eₖ(view(H, i, from:i-1), length(from:i-1))
        W₂ = Matrix{T}(I, m, m)
        W₂[from:i-1,from:i-1] .= I - 2 * (z₂ * z₂')
        Q .= Q * W₂
        H[1:m,1:m] .= W₂ * H[1:m,1:m] * W₂
    end

    for j = 1 : n, i = j + 2 : m
        H[i, j] = zero(T)
    end

    H[to+1,to] = H[n,m] * Q[m,to]
end