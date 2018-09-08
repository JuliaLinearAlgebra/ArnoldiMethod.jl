function localnorm(y::AbstractVector{T}, range) where {T}
    nrm = zero(real(T))
    @inbounds @simd for idx in range
        nrm += abs2(y[idx])
    end
    √nrm
end

"""
    reflector!(y, k) -> τ‖y‖

Computes z such that (I - 2zz') y = eₖν where ν = ‖y‖τ for some |τ| = 1. Updates y ← z 
in-place.

Based on Mezzadri, Francesco. "How to generate random matrices from the 
classical compact groups." arXiv preprint math-ph/0609050 (2006).
"""
function reflector!(y::AbstractVector{T}, k::Integer) where {T}
    @inbounds begin
        # The sign: |τ| = 1.
        τ = y[k] / abs(y[k])
        
        # Norm except the last entry.
        xnrm = localnorm(y, OneTo(k-1))

        # Total norm
        ynrm = hypot(xnrm, abs(y[k]))
        ν = τ * ynrm

        # y ← y + eₖ ‖y‖τ
        y[k] += ν

        # New y-norm
        yrnm_inv = inv(hypot(xnrm, abs(y[k])))
        
        # Rescale
        @simd for i = OneTo(k)
            y[i] *= yrnm_inv
        end

        return -ν
    end
end

struct Reflector{T}
    vec::Vector{T}
    offset::RefValue{Int}
    len::RefValue{Int}

    Reflector{T}(max_len::Int) where {T} = new{T}(
        Vector{T}(undef, max_len),
        Base.RefValue(1),
        Base.RefValue(0)
    )
end

function reflector!(z::Reflector, k::Integer)
    z.len[] = k
    return reflector!(z.vec, k)
end

"""
We have the Krylov relation

A[VQ₁ VQ₂] = [VQ₁ VQ₂][R₁₁ R₁₂; + hv[. eₖᵀQ₂]
                       .   R₂₂]

Column VQ₂ has column indices from:to
"""
function restore_hessenberg!(H::AbstractMatrix{T}, from::Integer, to::Integer, Q) where {T}
    m, n = size(H)

    @inbounds begin

        # Size of the initial reflector
        len = length(from:to)

        # Allocate a reflector of maximum length `len`; it will get smaller and smaller :)
        G = Reflector{T}(len)
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
        Q[n,to] = τ₁

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
            rmul!(H, G, 1, row)

            # Zero out things by hand
            @simd for j = OneTo(i - 1)
                H[row,j+from-1] = zero(T)
            end
            H[row, i-1+from] = τ₂

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

function lmul!(G::Reflector, H::AbstractMatrix{T}, from::Int, to::Int) where {T}
    @inbounds for col = from:to
        # H[i:j,col] ← H[i:j,col] - 2dot(G.vec[1:G.len], H[i:j,col]) G.vec[1:G.len]

        dot = zero(T)
        @simd for i = 1:G.len[]
            dot += G.vec[i]' * H[i+G.offset[]-1,col]
        end
        dot *= 2

        @simd for i = 1:G.len[]
            H[i+G.offset[]-1,col] -= dot * G.vec[i]
        end
    end
end

function rmul!(H::AbstractMatrix{T}, G::Reflector, from::Int, to::Int) where {T}
    @inbounds for row = from:to
        # H[row,i:j] ← H[row,i:j] - (2H[row,i:j]ᵀ G.vec[1:G.len]) G.vec[1:G.len]'

        dot = zero(T)
        @simd for i = 1:G.len[]
            dot += H[row,i+G.offset[]-1] * G.vec[i]
        end
        dot *= 2

        @simd for i = 1:G.len[]
            H[row,i+G.offset[]-1] -= dot * G.vec[i]'
        end
    end
end