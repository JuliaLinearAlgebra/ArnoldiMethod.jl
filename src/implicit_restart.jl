using Base.LinAlg: givensAlgorithm

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T<:Real}
    # Real arithmetic
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs, rev = true)
    Q = eye(T, max)

    callback = function(givens)
        # Apply the rotations to the G block
        mul!(view(H, 1:active-1, active:max), givens)
        
        # Apply the rotations to Q
        mul!(view(Q, :, active:max), givens)
    end

    m = max

    while m > min
        μ = λs[m-active+1]
        if imag(μ) == 0
            single_shift!(view(H, active : m + 1, active : m), real(μ), callback)
            m -= 1
        else
            double_shift!(view(H, active : m + 1, active : m), μ, callback)
            m -= 2
        end
    end

    # Update & copy the Krylov basis
    A_mul_B!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copy!(view(V, :, active:m), view(V_new, :, 1:m-active+1))
    copy!(view(V, :, m+1), view(V, :, max+1))

    return m
end

function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T}
    # Complex arithmetic
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs, rev = true)
    Q = eye(T, max)

    callback = function(givens)
        # Apply the rotations to the G block
        mul!(view(H, 1:active-1, active:max), givens)
        
        # Apply the rotations to Q
        mul!(view(Q, :, active:max), givens)
    end

    m = max

    while m > min
        μ = λs[m - active + 1]
        single_shift!(view(H, active : m + 1, active : m), μ, callback)
        m -= 1
    end

    # Update & copy the Krylov basis
    A_mul_B!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copy!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copy!(view(V, :, m + 1), view(V, :, max + 1))

    return m
end

"""
Assume a Hessenberg matrix of size (n + 1) × n.
"""
function single_shift!(H::AbstractMatrix, μ, callback = (x...) -> nothing; debug = false)
    @assert size(H, 1) == size(H, 2) + 1
    
    n = size(H, 2)

    # Construct the first givens rotation that maps (H - μI)e₁ to a multiple of e₁
    c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c, s, 1)

    mul!(givens, H)
    mul!(view(H, 1 : 3, :), givens)
    callback(givens)

    # Chase the bulge!
    for i = 2 : n - 1
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c, s, i)
        mul!(givens, view(H, 1 : i + 1, i-1:n))
        mul!(view(H, 1 : i + 2, :), givens)
        callback(givens)
    end

    # Do the last Given's rotation by hand (assuming exact shifts!)
    H[n, n - 1] = H[n + 1, n - 1]
    H[n + 1, n - 1] = zero(eltype(H))

    return H
end

function double_shift!(H::AbstractMatrix{Tv}, μ::Complex, callback = (x...) -> nothing; debug = false) where {Tv<:Real}
    @assert size(H, 1) == size(H, 2) + 1
    n = size(H, 2)

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    p₁ = abs2(μ) - 2 * real(μ) * H[1,1] + H[1,1] * H[1,1] + H[1,2] * H[2,1]
    p₂ = -2.0 * real(μ) * H[2,1] + H[2,1] * H[1,1] + H[2,2] * H[2,1]
    p₃ = H[3,2] * H[2,1]

    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, 2)
    G₂ = Givens(c₂, s₂, 1)

    mul!(G₁, H)
    mul!(G₂, H)
    mul!(H, G₁)
    mul!(H, G₂)

    callback(G₁)
    callback(G₂)

    # Bulge chasing!
    for i = 2 : n - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, i + 1)
        G₂ = Givens(c₂, s₂, i)

        # Restore to Hessenberg
        mul!(G₁, view(H, :, i-1:n))
        mul!(G₂, view(H, :, i-1:n))
        mul!(view(H, 1:i+3, :), G₁)
        mul!(view(H, 1:i+3, :), G₂)

        callback(G₁)
        callback(G₂)
    end

    # Do the last Given's rotation by hand.
    H[n - 1, n - 2] = H[n + 1, n - 2]
    H[n + 1, n - 2] = zero(Tv) # Not sure about this

    H
end
