using IRAM: Hessenberg, qr!, mul!, ListOfRotations
using Base.LinAlg: givensAlgorithm

struct Arnoldi{T}
    V::StridedMatrix{T}
    H::StridedMatrix{T}
end

struct PartialSchur{TQ,TR} 
    Q::TQ
    R::TR
    k::Int
end

"""
Allocate some space for an Arnoldi factorization of A where the Krylov subspace has
dimension `max` at most.
"""
function initialize(::Type{T}, n::Int, max = 30) where {T}
    V = Matrix{T}(n, max + 1)
    H = zeros(T, max + 1, max)

    v1 = view(V, :, 1)
    rand!(v1)
    v1 ./= norm(v1)

    Arnoldi(V, H)
end


"""
Perform Arnoldi iterations.
"""
function iterate_arnoldi!(A::AbstractMatrix{T}, arnoldi::Arnoldi{T}, range::UnitRange{Int}) where {T}
    V, H = arnoldi.V, arnoldi.H

    @inbounds @views for j = range
        v = V[:, j + 1]
        A_mul_B!(v, A, V[:, j])

        # Orthogonalize
        for i = 1 : j
            H[i, j] = dot(V[:, i], v)
            v .-= H[i, j] .* V[:, i]
        end

        # Normalize
        H[j + 1, j] = norm(v)
        v ./= H[j + 1, j]
    end

    return arnoldi
end

"""
Assume H is an (m + 1) x m unreduced Hessenberg matrix, the active part in the Arnoli
decomposition.
"""
function shifted_qr_step!(H::AbstractMatrix{T}, μ, rotations::ListOfRotations) where {T}
    m = size(H, 1) - 1

    # Apply the shift
    @inbounds for i = 1:m
        H[i,i] -= μ
    end

    # QR step; compute & apply Given's rotations from the left
    qr!(Hessenberg(view(H, 1:m, 1:m)), rotations)

    # Do the last Given's rotation by hand
    @inbounds H[m,m] = H[m+1,m]
    @inbounds H[m+1,m] = zero(T)
    
    # RQ step; apply Given's rotations from the right
    mul!(UpperTriangularMatrix(H), rotations)

    # Undo the shift
    @inbounds for i = 1:m
        H[i,i] += μ
    end
end

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T<:Real}
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs)
    Q = eye(T, max)

    callback = function(givens)
        # Apply the rotations to the G block
        mul!(H[1:active-1, active:max], givens)
        
        # Apply the rotations to Q
        mul!(Q[:, active:max], givens)
    end

    idx = 1
    m = max

    while m > min

        # Pick a shift
        μ = λs[idx]

        H_active = view(H, active : m, active : m)

        if imag(μ) == 0
            # Single
            single_shift!(H_active, real(μ), callback)
            idx += 1
            m -= 1
        else
            # Double
            double_shift!(H_active, μ, callback)
            idx += 2
            m -= 2
        end
    end

    # Update the Krylov basis
    A_mul_B!(view(V_new,:,1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))

    # Copy to the Arnoldi factorization
    copy!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copy!(view(V, :, m + 1), view(V, :, max + 1))

    return arnoldi, m
end

function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T}
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs)
    Q = eye(T, max)

    callback = function(givens)
        # Apply the rotations to the G block
        mul!(H[1:active-1, active:max], givens)
        
        # Apply the rotations to Q
        mul!(Q[:, active:max], givens)
    end

    idx = 1
    m = max

    while m > idx

        # Pick a shift
        μ = λs[idx]

        H_active = view(H, active : m, active : m)

        # Single
        single_shift!(H_active, real(μ), callback)

        H[m + 1, m]
        idx += 1
        m -= 1
    end

    # Update the Krylov basis
    A_mul_B!(view(V_new,:,1:m-active+1), view(V, :, active:max), view(Q, active:max, active:m))
    
    # Copy to the Arnoldi factorization
    copy!(view(V, :, active:m), view(V_new,:,1:m-active+1))
    copy!(view(V, :, m + 1), view(V, :, max + 1))

    return arnoldi, m
end


"""
Find the largest index `i` such that `H[i, i-1] ≈ 0`. This means that `i:max` constitutes the
active part in the Arnoldi decomp, while `V[:, 1:i-1]` forms a basis for an invariant 
subspace of A. Sets H[i, i-1] := 0.
"""
function detect_convergence!(H::AbstractMatrix{T}, tolerance) where {T}
    @inbounds for i = size(H, 2) : -1 : 2
        if abs(H[i, i-1]) ≤ tolerance
            H[i, i-1] = zero(T)
            return i
        end
    end

    # No convergence :(
    return 1
end

"""
Run IRAM until the eigenpair is a good enough approximation or until max_restarts has been reached
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, converged = min, tolerance = 1e-5, max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : min)

    active = 1
    V_new = Matrix{T}(n,min)

    for restarts = 1 : max_restarts
        iterate_arnoldi!(A, arnoldi, min + 1 : max)
        implicit_restart!(arnoldi, min, max, active, V_new)

        new_active = detect_convergence!(view(arnoldi.H, 1:min+1, 1:min), tolerance)

        # Bring the new locked part oF H into upper triangular form
        if new_active > active + 1
            schur_form = schur(view(arnoldi.H, active : new_active - 1, active : new_active - 1))
            arnoldi.H[active : new_active - 1, active : new_active - 1] .= schur_form[1]

            V_locked = view(arnoldi.V, :, active : new_active - 1)
            H_right = view(arnoldi.H, active : new_active - 1, new_active : min)

            A_mul_B!(V_locked, copy(V_locked), schur_form[2])
            Ac_mul_B!(H_right, schur_form[2], copy(H_right))
            
            if active > 1 
                H_above = view(arnoldi.H, 1 : active - 1, active : new_active - 1)
                A_mul_B!(H_above, copy(H_above), schur_form[2])
            end
        end

        active = new_active

        @show active

        if active >= converged
            break 
        end
    end
end

function single_shift!(H::AbstractMatrix, μ, callback = (x...) -> nothing, debug = false)

    n = size(H,2)

    if debug
        println("Initial H")
        display("text/plain", H)
        println()
        println("Compute Givens rotation that maps the first column of H - μI to r₁₁q₁: G₁ * H")
    end

    c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c,s,1)

    mul!(givens, Hessenberg(H))
    callback(givens)

    display("text/plain", H)
    println()

    println("Apply the Givens rotation from the rhs")

    # Apply the rotation from the right.
    mul!(H, givens)

    display("text/plain", H)

    # Restore to Hessenberg form
    for i = 2 : n - 1
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c, s, i)
        
        mul!(givens, view(H, 1 : i + 1, i-1:n)) # Assumes that H is (n+1)*n
        
        if debug
            println("Restore Hessenberg form")
            display("text/plain", H)
            println()
        end
        
        last_row = min(i + 2, n)
        mul!(view(H, 1 : last_row, :), givens)

        callback(givens)

        if debug
            println("Apply from the rhs")
            display("text/plain", H)
            println()
        end
    end
end

function double_shift!(H, μ::Complex, callback = (x...) -> nothing, debug = true)
    @show size(H, 2) size(H, 1)
    @assert size(H, 2) == size(H, 1)

    n = size(H, 1)

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    poly11 = abs2(μ) - 2 * real(μ) * H[1,1] + H[1,1] * H[1,1] + H[1,2] * H[2,1]
    poly12 = - 2.0 * real(μ) * H[2,1] + H[2,1] * H[1,1] + H[2,2] * H[2,1]
    poly13 = H[3,2] * H[2,1]

    # Sanity check
    # @assert fst ≈ ((H - μ * I) * ((H - conj(μ) * I)[:, 1]))[1:3]

    c1, s1, el = givensAlgorithm(poly12, poly13)
    c2, s2 = givensAlgorithm(poly11, el)
    givens_first = Givens(c1, s1, 2)
    givens_second = Givens(c2, s2, 1)

    mul!(givens_first, H)
    mul!(givens_second, H)
    
    if debug
        println("Compute Householder reflection that maps G₁ * (H - μ₂)(H - μ₁)e₁ = r₁₁e₁")
        display("text/plain", H)
        println()
    end

    # H *= G' # and apply the rotation from the right.
    mul!(H, givens_first)
    mul!(H, givens_second)

    callback(givens_first)
    callback(givens_second)

    if debug
        println("Apply the Housholder reflection from the rhs: G₁ * H * G₁'")
        display("text/plain", H)
        str = "G₁ * H * G₁'"
    end

    # Restore to Hessenberg form
    # Bulge is initially in H[1:3,1:3]
    for i = 2 : n - 2
        c1, s1, el = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c2, s2 = givensAlgorithm(H[i,i-1], el)
        givens_first = Givens(c1, s1, i + 1)
        givens_second = Givens(c2, s2, i)

        H_bulge_block = view(H, 1 : i + 2, i - 1 : n)
        mul!(givens_first, H_bulge_block)
        mul!(givens_second, H_bulge_block)

        if debug
            println("\n------\n\n")
            println("Restore Hessenberg form: $str")
            display("text/plain", H)
            println()
        end

        # Destroy
        last_row = min(i + 3, n)
        H_other_bulge_block = view(H, 1 : last_row, 1 : i + 2)
        mul!(H_other_bulge_block, givens_first)
        mul!(H_other_bulge_block, givens_second)

        if debug
            println("Apply from the rhs: $str")
            display("text/plain", H)
            println()
        end

        callback(givens_first)
        callback(givens_second)
    end
end

function qr_callback(active, m, givens)
    
    # Apply the rotations to the G block
    mul!(view(H,1:active-1, active:m), givens)

    # Apply the rotations to Q
    mul!(view(Q,1:max, active:m), givens)

end

function implicit_qr_step!(H::Hessenberg, active, max, L::ListOfRotations, tolerance)
    
    λs = sort!(eigvals(view(H.H, active:max, active:max)), by = abs) #Determine if single or double shift
    # λs[1] 
    single_shift!(H.H, λs[1], L, qr_callback) 
    # active = detect_convergence!(view(H.H, 1:min+1, 1:min), tolerance) # Check for deflation
    
    return (active, max)

end