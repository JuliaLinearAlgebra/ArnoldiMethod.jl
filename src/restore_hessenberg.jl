"""
    reflector!(y, k) → τ

Implicit representation of a Householder reflection H = I - τ [v; 1] [vᵀ 1]
such that H*y = eₖβ, 1 ≤ real(τ) ≤ 2 and abs(τ - 1) ≤ 1. Updates `y` in-place,
in the sense that: `v := y[1:k-1]` and y[k] = β.

In the edge case iszero(y[1:k-1]) and iszero(imag(y[k])) then τ = 0.

Based on LAPACK 3.8 clarfg.

Another nice reference is Mezzadri, Francesco. "How to generate
random matrices from the classical compact groups." arXiv preprint 
math-ph/0609050 (2006).
"""
function reflector!(y::AbstractVector{T}, k::Integer) where {T}
    @inbounds begin
        k ≤ 0 || k > length(y) && return zero(T)

        # Norm except the last entry.
        xnrm = zero(real(T))
        @simd for idx in OneTo(k-1)
            xnrm += abs2(y[idx])
        end

        α = y[k]

        iszero(xnrm) && iszero(imag(α)) && return zero(T)

        xnrm = √xnrm

        β = -copysign(hypot(α, xnrm), real(α))
        τ = (β - α) / β
        α = inv(α - β)
       
        # Rescale
        @simd for i = OneTo(k-1)
            y[i] *= α
        end

        y[k] = β

        return τ'
    end
end


function reflector!(z::Reflector, k::Integer)
    z.len[] = k
    z.τ[] = reflector!(z.vec, k)
    return z.τ[];
end

"""
We have the Krylov relation

A[VQ₁ VQ₂] = [VQ₁ VQ₂][R₁₁ R₁₂; + hv[. eₖᵀQ₂]
                       .   R₂₂]

Column VQ₂ has column indices from:to
"""
function restore_arnoldi!(H::AbstractMatrix{T}, from::Integer, to::Integer, Q, G::Reflector{T}) where {T}
    m, n = size(H)

    @inbounds begin

        # Size of the initial reflector
        len = length(from:to)

        len ≤ 2 && return nothing

        G.offset[] = from

        # Copy over the last row of Q
        @simd for i = OneTo(len)
            G.vec[i] = Q[n,i+from-1]'
        end

        # Zero out entries 1:len-1
        τ₁ = reflector!(G, len)

        # Apply to Q from the right.
        rmul!(Q, G, 1, n - 1)

        # Handle the last row by hand
        @simd for i = from:to-1
            Q[n,i] = zero(T)
        end
        Q[n,to] = G.vec[len]'

        # Then apply to H from both sides.
        rmul!(H, G, 1, to)
        lmul!(G, H, from, to)

        # Then restore the Hessenberg structure in H, which is now a full matrix.
        for i = len-1:-1:2
            row = from + i

            # Copy over row `row`
            @simd for j = OneTo(i)
                G.vec[j] = H[row,j+from-1]'
            end

            # Construct a reflector from it
            τ₂ = reflector!(G, i)

            # Apply it to the right to H
            rmul!(H, G, 1, row-1)

            # Zero out things by hand
            @simd for j = OneTo(i - 1)
                H[row,j+from-1] = zero(T)
            end
            H[row, i-1+from] = G.vec[i]'

            # Apply if from the left.
            lmul!(G, H, from, to)

            # Accumulate the reflectors
            rmul!(Q, G, 1, n)
        end

        # Finally in the Arnoldi decomp we want a last residual term of the form
        # h * vₖ₊₁ * eₖᵀ, so we absorb the last Q-entry in the Hessenberg matrix!
        H[to+1,to] = Q[end,to] * H[m, n]

    end

    nothing
end

# Implementations are terrible here :(

function lmul!(G::Reflector, H::AbstractMatrix{T}, from::Int, to::Int) where {T}

    len = G.len[]
    offset = G.offset[]
    z = G.vec
    τ = G.τ[]

    iszero(τ) && return nothing

    @inbounds for col = from:to
        dot = zero(T)
        @simd for i = 1 : len - 1
            dot += z[i]' * H[i + offset - 1, col]
        end
        dot += H[len + offset - 1, col]
        dot *= τ
        @simd for i = 1 : len - 1
            H[i + offset - 1, col] -= dot * z[i]
        end
        H[len + offset - 1, col] -= dot
    end
end

function rmul!(H::AbstractMatrix{T}, G::Reflector, from::Int, to::Int) where {T}

    len = G.len[]
    offset = G.offset[]
    z = G.vec
    τ = G.τ[]

    iszero(τ) && return nothing

    @inbounds for row = from:to
        dot = zero(T)
        @simd for i = 1:len-1
            dot += H[row, i + offset - 1] * z[i]
        end
        dot += H[row, offset + len - 1]
        dot *= τ'
        @simd for i = 1 : len - 1
            H[row,i+offset-1] -= dot * z[i]'
        end
        H[row, offset + len - 1] -= dot
    end
end