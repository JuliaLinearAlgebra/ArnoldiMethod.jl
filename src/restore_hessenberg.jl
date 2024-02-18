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
        @simd for idx in OneTo(k - 1)
            xnrm += abs2(y[idx])
        end

        α = y[k]

        iszero(xnrm) && iszero(imag(α)) && return zero(T)

        xnrm = √xnrm

        β = -copysign(hypot(α, xnrm), real(α))
        τ = (β - α) / β
        α = inv(α - β)

        # Rescale
        @simd for i in OneTo(k - 1)
            y[i] *= α
        end

        y[k] = β

        return τ'
    end
end

struct Reflector{T}
    vec::Vector{T}
    offset::RefValue{Int}
    len::RefValue{Int}
    τ::RefValue{T}

    Reflector{T}(max_len::Int) where {T} = new{T}(
        Vector{T}(undef, max_len),
        Base.RefValue(1),
        Base.RefValue(0),
        Base.RefValue(zero(T)),
    )
end

function reflector!(z::Reflector, k::Integer)
    z.len[] = k
    z.τ[] = reflector!(z.vec, k)
    return z.τ[]
end

"""
We have the Krylov relation

A[VQ₁ VQ₂] = [VQ₁ VQ₂][R₁₁ R₁₂; + hv[. eₖᵀQ₂]
                       .   R₂₂]

Column VQ₂ has column indices from:to
"""
function restore_arnoldi!(
    H::AbstractMatrix{T},
    from::Integer,
    to::Integer,
    Q,
    G::Reflector{T},
) where {T}
    from < to || return nothing

    m, n = size(H)

    # Use Given's rotations to zero out the last column of Q, which makes H[from:to, from:to]
    # a dense block. We use rotations because it's more stable, since the Q values are residual
    # norms, with varying orders of magnitude.
    @inbounds nrm = Q[n, from]
    @inbounds for i = from:to-1
        c, s, nrm = givensAlgorithm(Q[n, i+1], nrm)
        givens_rotation = Rotation2(c, -s, i)
        rmul!(H, givens_rotation, 1, min(i + 2, to))
        lmul!(givens_rotation, H, 1, to)
        rmul!(Q, givens_rotation, 1, n)
    end

    # In the Arnoldi decomp we want a last residual term of the form h * vₖ₊₁ * eₖᵀ, so we absorb
    # it in H.
    @inbounds H[to+1, to] = Q[end, to] * H[m, n]

    G.offset[] = from

    # Then restore the Hessenberg structure in H, which is now a full matrix.
    @inbounds for i = to-from:-1:2
        G.len[] = i
        row = from + i

        # Copy over row `row`
        @simd for j in OneTo(i)
            G.vec[j] = H[row, j+from-1]'
        end

        # Construct a reflector from it
        reflector!(G, i)

        # Apply it to the right to H
        rmul!(H, G, 1, row - 1)

        # Zero out things by hand
        @simd for j in OneTo(i - 1)
            H[row, j+from-1] = zero(T)
        end
        H[row, i-1+from] = G.vec[i]'

        # Apply if from the left.
        lmul!(G, H, from, to)

        # Accumulate the reflectors
        rmul!(Q, G, 1, n)
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
        @simd for i = 1:len-1
            dot += z[i]' * H[i+offset-1, col]
        end
        dot += H[len+offset-1, col]
        dot *= τ
        @simd for i = 1:len-1
            H[i+offset-1, col] -= dot * z[i]
        end
        H[len+offset-1, col] -= dot
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
            dot += H[row, i+offset-1] * z[i]
        end
        dot += H[row, offset+len-1]
        dot *= τ'
        @simd for i = 1:len-1
            H[row, i+offset-1] -= dot * z[i]'
        end
        H[row, offset+len-1] -= dot
    end
end
