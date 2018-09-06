using ArnoldiMethod: NotWanted, Rotation2, get_rotation
using LinearAlgebra

"""
[λ α] → [μ β]
[0 μ] → [0 λ]
"""
function swap_rotation(λ, μ, α)
    t = (μ - λ) / α
    c = 1 / √(1 + abs(t) ^ 2)
    s = c * t
    c, s
end

"""
Move R[from,from] to S[to,to] using Givens rotations
"""
function swap_schur!(R::AbstractMatrix, from::Int, to::Int, Q = NotWanted())
    m, n = size(R)

    @inbounds for l = from-1:-1:to
        R11 = R[l+0,l+0]
        R12 = R[l+0,l+1]
        R22 = R[l+1,l+1]
        
        G, nrm = get_rotation(R12, R22 - R11, l)
        
        # Swap S[l, l] and S[l + 1, l + 1]
        lmul!(G, R, l+2, n)
        rmul!(R, G, 1, l-1)
        R[l+0,l+0] = R22
        R[l+1,l+1] = R11

        # Accumulate Q
        rmul!(Q, G)
    end
end

function example(n = 10)
    # Generate a Schur decomp
    R = triu(rand(n, n))
    Q = qr(rand(n, n)).Q * Matrix(I, n, n)
    A = Q * R * Q'

    Q2 = copy(Q)
    R2 = copy(R)

    swap_schur!(R2, 10, 1, Q2)

    return (A,Q,R), (A,Q2,R2)
end