using LinearAlgebra: givensAlgorithm

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(undef, size(arnoldi.V,1),min)) where {T<:Real}
    # Real arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max, max)

    m = max

    while m > min
        μ = λs[m-active+1]
        if imag(μ) == 0
            single_shift!(H, active, m, real(μ), Q)
            m -= 1
        else
            # Dont double shift past min
            # m == min + 1 && break

            double_shift!(H, active, m, μ, Q)
            m -= 2 # incorrect
        end
    end

    # Update & copy the Krylov basis
    mul!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copyto!(view(V, :, active:m), view(V_new, :, 1:m-active+1))
    copyto!(view(V, :, m+1), view(V, :, max+1))

    return m
end

function implicit_restart!(arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(undef, size(arnoldi.V,1),min)) where {T}
    # Complex arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max, max)

    m = max

    while m > min
        μ = λs[m - active + 1]
        single_shift!(H, active, m, μ, Q)
        m -= 1
    end

    # Update & copy the Krylov basis
    mul!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copyto!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copyto!(view(V, :, m + 1), view(V, :, max + 1))

    return m
end

# """
# Assume a Hessenberg matrix of size (n + 1) × n.
# """
function single_shift!(H_whole::AbstractMatrix{Tv}, min, max, μ::Tv, Q::AbstractMatrix; debug = false) where {Tv}
    # println("Single:")
    H = view(H_whole, min : max + 1, min : max)
    n = size(H, 2)

    # Construct the first givens rotation that maps (H - μI)e₁ to a multiple of e₁
    c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c, s, min)

    lmul!(givens, H_whole)
    rmul!(H_whole, givens)

    # Update Q
    rmul!(Q, givens)

    # Chase the bulge!
    for i = 2 : n - 1
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c, s, min + i - 1)
        lmul!(givens, H_whole)
        H_whole[i+1,i-1] = zero(Tv)
        rmul!(view(H_whole, 1 : min + i + 1, :), givens)
        
        # Update Q
        rmul!(Q, givens)
    end

    # Do the last Given's rotation by hand (assuming exact shifts!)
    H[n, n - 1] = H[n + 1, n - 1]
    H[n + 1, n - 1] = zero(eltype(H))

    return H
end

function double_shift!(H_whole::AbstractMatrix{Tv}, min, max, μ::Complex, Q::AbstractMatrix; debug = false) where {Tv<:Real}
    H = view(H_whole, min : max + 1, min : max)
    n = size(H, 2)

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    p₁ = abs2(μ) - 2 * real(μ) * H[1,1] + H[1,1] * H[1,1] + H[1,2] * H[2,1]
    p₂ = -2.0 * real(μ) * H[2,1] + H[2,1] * H[1,1] + H[2,2] * H[2,1]
    p₃ = H[3,2] * H[2,1]

    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, min+1)
    G₂ = Givens(c₂, s₂, min)

    lmul!(G₁, H_whole)
    lmul!(G₂, H_whole)
    rmul!(H_whole, G₁)
    rmul!(H_whole, G₂)

    # Update Q
    rmul!(Q, G₁)
    rmul!(Q, G₂)

    # Bulge chasing!
    for i = 2 : n - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, min+i)
        G₂ = Givens(c₂, s₂, min+i-1)

        # Restore to Hessenberg
        lmul!(G₁, view(H_whole, :, min+i-2:max))
        lmul!(G₂, view(H_whole, :, min+i-2:max))
        
        # Zero out off-diagonal values
        H_whole[min+i, i-1] = zero(Tv)
        H_whole[min+i+1, i-1] = zero(Tv)

        # Create a new bulge
        rmul!(view(H_whole, 1:min+i+2, :), G₁)
        rmul!(view(H_whole, 1:min+i+2, :), G₂)

        # Update Q
        rmul!(Q, G₁)
        rmul!(Q, G₂)
    end

    if n > 2
        # Do the last Given's rotation by hand.
        H[n - 1, n - 2] = H[n + 1, n - 2]

        # Zero out the off-diagonal guys
        H[n    , n - 2] = zero(Tv)
        H[n + 1, n - 2] = zero(Tv)
    end

    H[n + 1, n - 1] = zero(Tv)
    
    H
end
