using LinearAlgebra: givensAlgorithm

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, λs, min = 5, max = 30, active = 1, V_new = Matrix{T}(undef, size(arnoldi.V,1),min)) where {T<:Real}
    # Real arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max+1, max+1)

    m = max

    while m > min
        μ = λs[m-active+1]
        if imag(μ) == 0
            exact_single_shift!(H, active, m, real(μ), Q)
            m -= 1
        else
            # Dont double shift past min
            # m == min + 1 && break

            exact_double_shift!(H, active, m, μ, Q)
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
    # General arithmetic
    V, H = arnoldi.V, arnoldi.H
    Q = Matrix{T}(I, max+1, max+1)

    m = max

    while m > min
        μ = λs[m - active + 1]
        exact_single_shift!(H, active, m, μ, Q)
        m -= 1
    end

    # Update & copy the Krylov basis
    mul!(view(V_new, :, 1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    copyto!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copyto!(view(V, :, m + 1), view(V, :, max + 1))

    return m
end

"""
    exact_single_shift!(H, min, max, μ, Q)

Performs an exact single shift of the QR algorithm on the non-square Hessenberg
matrix H[from:to+1,from:to]
"""
function exact_single_shift!(H::AbstractMatrix{Tv}, from::Int, to::Int, μ::Number, Q) where {Tv}
    # Construct the first givens rotation that maps (H - μI)e₁ to a multiple of e₁
    @inbounds H₁₁ = H[from+0,from+0]
    @inbounds H₂₁ = H[from+1,from+0]

    p₁ = H₁₁ - μ
    p₂ = H₂₁

    G₁, nrm = get_rotation(p₁, p₂, from)

    lmul!(G₁, H, from, to)
    rmul!(H, G₁, 1, min(from + 2, to + 1))
    rmul!(Q, G₁)

    # Chase the bulge!
    @inbounds for i = from + 1 : to - 1
        p₁ = H[i+0,i-1]
        p₂ = H[i+1,i-1]

        G, nrm = get_rotation(p₁, p₂, i)

        # First column is done by hand
        H[i+0,i-1] = nrm
        H[i+1,i-1] = zero(Tv)
        
        # Rotate remaining columns
        lmul!(G, H, i, to)

        # Create a new bulge
        rmul!(H, G, 1, min(i + 2, to + 1))
        rmul!(Q, G)
    end

    # Do the last Given's rotation by hand (assuming exact shifts!)
    @inbounds H[to+0,to-1] = H[to+1,to-1]
    @inbounds H[to+1,to-1] = zero(Tv)

    # Update Q with the last rotation
    Q[1:max+1, max] .= 0
    Q[max+1,max] = 1
    
    return H
end

"""
    exact_double_shift!(H, min, max, μ, Q)

Performs an exact double shift of the QR algorithm on the non-square Hessenberg
matrix H[from:to+1,from:to]
"""
function exact_double_shift!(H::AbstractMatrix{Tv}, from::Int, to::Int, μ::Complex, Q::AbstractMatrix) where {Tv<:Real}
    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    @inbounds H₁₁ = H[from+0,from+0]
    @inbounds H₂₁ = H[from+1,from+0]

    @inbounds H₁₂ = H[from+0,from+1]
    @inbounds H₂₂ = H[from+1,from+1]
    @inbounds H₃₂ = H[from+2,from+1]

    p₁ = abs2(μ) - 2real(μ) * H₁₁ + H₁₁ * H₁₁ + H₁₂ * H₂₁
    p₂ = -2real(μ) * H₂₁ + H₂₁ * H₁₁ + H₂₂ * H₂₁
    p₃ = H₃₂ * H₂₁

    # Map that column to a mulitiple of e₁ via two Given's rotations
    G₁, nrm = get_rotation(p₁, p₂, p₃, from)

    # Apply the Given's rotations
    lmul!(G₁, H, from, to)
    rmul!(H, G₁, 1, min(from + 3, to + 1))
    rmul!(Q, G₁)

    @inbounds for i = from + 1 : to - 2
        p₁ = H[i+0,i-1]
        p₂ = H[i+1,i-1]
        p₃ = H[i+2,i-1]

        G, nrm = get_rotation(p₁, p₂, p₃, i)

        # First column is done by hand
        H[i+0,i-1] = nrm
        H[i+1,i-1] = zero(Tv)
        H[i+2,i-1] = zero(Tv)
        
        # Rotate remaining columns
        lmul!(G, H, i, to)

        # Create a new bulge
        rmul!(H, G, 1, min(i + 3, to + 1))
        rmul!(Q, G)
    end

    # Do the last Given's rotation by hand.
    @inbounds H[to-1,to-2] = H[to+1,to-2]

    # Zero out the off-diagonal guys
    @inbounds H[to+0,to-2] = zero(Tv)
    @inbounds H[to+1,to-2] = zero(Tv)
    @inbounds H[to+1,to-1] = zero(Tv)
    
    H
end
