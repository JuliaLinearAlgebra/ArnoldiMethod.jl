using Base.LinAlg: givensAlgorithm

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T<:Real}
    # Real arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = eye(T, max)

    # callback = function(givens)
    #     # Apply the rotations to the G block
    #     mul!(view(H, 1:active-1, active:max), givens)
        
    #     # Apply the rotations to Q
    #     mul!(view(Q, :, active:max), givens)
    # end

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
    A_mul_B!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copy!(view(V, :, active:m), view(V_new, :, 1:m-active+1))
    copy!(view(V, :, m+1), view(V, :, max+1))

    return m
end

function implicit_restart!(arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T}
    # Complex arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = eye(T, max)

    # callback = function(givens)
    #     # Apply the rotations to the G block
    #     mul!(view(H, 1:active-1, active:max), givens)
        
    #     # Apply the rotations to Q
    #     mul!(view(Q, :, 1:max), givens)
    # end

    m = max

    while m > min
        μ = λs[m - active + 1]
        single_shift!(H, active, m, μ, Q)
        m -= 1
    end

    # Update & copy the Krylov basis
    A_mul_B!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copy!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copy!(view(V, :, m + 1), view(V, :, max + 1))

    return m
end

# """
# Assume a Hessenberg matrix of size (n + 1) × n.
# """
function single_shift!(H_whole::AbstractMatrix, min, max, μ, Q::AbstractMatrix; debug = false)
    # println("Single:")
    H = view(H_whole, min : max + 1, min : max)
    # @assert size(H, 1) == size(H, 2) + 1
    
    n = size(H, 2)

    # Construct the first givens rotation that maps (H - μI)e₁ to a multiple of e₁
    c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c, s, min)

    mul!(givens, H_whole)
    mul!(H_whole, givens)

    # Update Q
    mul!(Q, givens)

    # Chase the bulge!
    for i = 2 : n - 1
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c, s, min + i - 1)
        mul!(givens, H_whole)
        mul!(view(H_whole, 1 : min + i + 1, :), givens)
        
        # Update Q
        mul!(Q, givens)
    end

    # Do the last Given's rotation by hand (assuming exact shifts!)
    H[n, n - 1] = H[n + 1, n - 1]
    H[n + 1, n - 1] = zero(eltype(H))

    return H
end

function double_shift!(H_whole::AbstractMatrix{Tv}, min, max, μ::Complex, Q::AbstractMatrix; debug = false) where {Tv<:Real}
    # println("Double:")
    H = view(H_whole, min : max + 1, min : max)
    # @assert size(H, 1) == size(H, 2) + 1
    n = size(H, 2)

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    p₁ = abs2(μ) - 2 * real(μ) * H[1,1] + H[1,1] * H[1,1] + H[1,2] * H[2,1]
    p₂ = -2.0 * real(μ) * H[2,1] + H[2,1] * H[1,1] + H[2,2] * H[2,1]
    p₃ = H[3,2] * H[2,1]

    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, min+1)
    G₂ = Givens(c₂, s₂, min)

    mul!(G₁, H_whole)
    mul!(G₂, H_whole)
    mul!(H_whole, G₁)
    mul!(H_whole, G₂)

    # Update Q
    mul!(Q, G₁)
    mul!(Q, G₂)

    # callback(G₁)
    # callback(G₂)

    # Bulge chasing!
    for i = 2 : n - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, min+i)
        G₂ = Givens(c₂, s₂, min+i-1)

        # Restore to Hessenberg
        mul!(G₁, view(H_whole, :, min+i-2:max))
        mul!(G₂, view(H_whole, :, min+i-2:max))
        mul!(view(H_whole, 1:min+i+2, :), G₁)
        mul!(view(H_whole, 1:min+i+2, :), G₂)

        # Update Q
        mul!(Q, G₁)
        mul!(Q, G₂)
    
        # callback(G₁)
        # callback(G₂)
    end

    # Do the last Given's rotation by hand.
    H[n - 1, n - 2] = H[n + 1, n - 2]

    H[n + 1, n - 2] = zero(Tv) # Not sure about these
    #H[n + 1, n - 1] = zero(Tv)

    #display(H_whole[1:4,1:4]); println()

    H
end
