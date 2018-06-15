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
function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1, V_new = Matrix{T}(size(arnoldi.V,1),min)) where {T}
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs)
    Q = eye(T, max)
    rotations = ListOfRotations(T, max)

    idx = 1

    @views for m = max : -1 : min + 1

        # Pick a shift
        μ = λs[idx]

        # Apply a shifted QR step to the H[active:m, active:m]
        shifted_qr_step!(H[active:m+1, active:m], μ, rotations)

        # Apply the rotations to the G block
        mul!(H[1:active-1, active:m], rotations)
        
        # Apply the rotations to Q
        mul!(Q[1:max, active:m], rotations)

        idx += 1
    end

    # Update the Krylov basis
    A_mul_B!(view(V_new,:,1:min-active+1), view(V, :, active:max), view(Q, active:max, active:min))
    
    # Update the Krylov basis
    # V_new = view(V, :, active:max) * view(Q, active:max, active:min)

    # Copy to the Arnoldi factorization
    copy!(view(V, :, active:min), view(V_new,:,1:min-active+1))
    copy!(view(V, :, min + 1), view(V, :, max + 1))

    return arnoldi
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
            arnoldi.H[active : new_active - 1, active : new_active - 1] = schur_form[1]

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

    return PartialSchur(arnoldi.V, arnoldi.H, active - 1)
end

function single_shift!(H::AbstractMatrix, μ, L::ListOfRotations, callback = (x...) -> nothing)

    n = size(H,2)

    indices = collect("₁₂₃₄₅₆₇₈₉")

    println("Initial H")
    # H = view(H.H, :, :)
    display("text/plain", H)
    println()

    println("Compute Givens rotation that maps the first column of H - μI to r₁₁q₁: G₁ * H")
    c, s = givensAlgorithm(H[1,1] - μ, H[2,1])
    givens = Givens(c,s,1)

    mul!(givens,Hessenberg(H))

    display("text/plain", H)
    println()

    println("Apply the Givens rotation from the rhs: G₁ * H * G₁'")

    # Apply the rotation from the right.
    mul!(H, givens)

    display("text/plain", H)

    str = "G₁ * H * G₁'"

    # Restore to Hessenberg form
    for i = 2 : n - 1
        println("\n\n------\n\n")
        str = "G$(indices[i]) * " * str
        
        c, s = givensAlgorithm(H[i,i-1], H[i+1,i-1])
        givens = Givens(c,s,i-1)
        
        println("Restore Hessenberg form: $str")
        mul!(givens,Hessenberg(view(H,2:n+1,:))) # Assumes that H is (n+1)*n
        display("text/plain", H)
        println()

        str = str * " * G$(indices[i])'"
        println("Apply from the rhs: $str")
        
        mul!(view(H,:,2:n),givens)
        display("text/plain", H)
        println()
    end
    
    # Callback, if double shift then callback twice
    # callback()
    
    return H

end

function double_shift!(H_original, μ₁, μ₂, debug = true, callback = (x...) -> nothing)
    @assert size(H_original, 2) == size(H_original, 1)

    n = size(H_original, 1)
    indices = collect("₁₂₃₄₅₆₇₈₉")
    H = copy(complex(H_original))
    Q = eye(Complex128, n)

    if debug
        println("Initial H")
        display("text/plain", H)
        println()
    end

    # α, β = H[n-1, n-1], H[n-1, n]
    # γ, δ = H[n, n-1],   H[n, n]
    # tr = α + δ
    # det = α * δ - γ * β
    # D = tr^2 - 4det

    # if D < 0
    #     return
    # end

    # μ₁ = tr + √D

    # μ₁ = 1.10
    # μ₂ = 1.4

    # Compute the entries of (H - μ₂)(H - μ₁)e₁.
    # Please don't do it like this :p, we should not store the household reflection as a
    # vector -- if we would wanna store it as a vector, then we should use SVector from 
    # StaticArrays.jl; but yeah, 2 givens rotations work OK as well
    fst = H[1:3, 1] # H * e₁
    fst[1] -= μ₁ # - μ₁ * e₁
    fst .= H[1:3, 1:2] * fst[1:2] - μ₂ * fst # well...

    # Sanity check
    # @assert fst ≈ ((H - μ₂ * I) * ((H - μ₁ * I)[:, 1]))[1:3]

    # G = eye(n)
    # G[1:3,1:3] .= convert(Matrix, householder(fst))
    # H = G * H

    c1, s1, el = givensAlgorithm(fst[2], fst[3])
    c2, s2 = givensAlgorithm(fst[1], el)
    givens_first = Givens(c1,s1,1)
    givens_second = Givens(c2,s2,1)

    mul!(givens_first, Hessenberg(view(H,2:n,:)))
    mul!(givens_second, Hessenberg(view(H,1:n,:)))
    
    if debug
        println("Compute Householder reflection that maps G₁ * (H - μ₂)(H - μ₁)e₁ = r₁₁e₁")
        display("text/plain", H)
        println()
    end

    # H *= G' # and apply the rotation from the right.
    # Q *= G' # callback, somehow
    mul!(view(H,:,2:n), givens_first)
    mul!(view(H,:,1:n), givens_second)
    # Q *= G'  # callback, somehow
    mul!(Q, givens_first)
    mul!(Q, givens_second)

    if debug
        println("Apply the Housholder reflection from the rhs: G₁ * H * G₁'")
        display("text/plain", H)
        str = "G₁ * H * G₁'"
    end

    # Restore to Hessenberg form
    # Bulge is initially in H[1:3,1:3]
    for i = 2 : n - 2

        # Restore...
        # range = i : min(i + 2, n) # when i = n - 1, this is just a given's rotation
        # G = eye(n)
        # G[range,range] .= convert(Matrix, householder(H[range, i - 1]))
        # H = G * H

        c1, s1, el = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c2, s2 = givensAlgorithm(H[i,i-1], el)
        givens_first = Givens(c1,s1,i-1)
        givens_second = Givens(c2,s2,i-1)

        mul!(givens_first, Hessenberg(view(H,3:n,:)))
        mul!(givens_second, Hessenberg(view(H,2:n,:)))

        if debug
            println("\n------\n\n")
            str = "G$(indices[i]) * " * str
            println("Restore Hessenberg form: $str")
            display("text/plain", H)
            println()
        end

        # Destroy
        # H *= G'
        mul!(view(H,:,3:n), givens_first)
        mul!(view(H,:,2:n), givens_second)
        # Q *= G'  # callback, somehow
        # mul!(Q, givens_first)
        # mul!(Q, givens_second)

        if debug
            str = str * " * G$(indices[i])'"
            println("Apply from the rhs: $str")
            display("text/plain", H)
            println()
        end
    end

    #Not entirely sure about the indexing here # EDIT: Seems about right
    # c1, s1 = givensAlgorithm(H[n-1,n-2], H[n,n-2])
    # givens_first = Givens(c1,s1,n-2)
    # mul!(givens_first, Hessenberg(view(H,3:n,:)))
    # if debug
    #     println("\n------\n\n")
    #     str = "G$(indices[n-1]) * " * str
    #     println("Restore Hessenberg form: $str")
    #     display("text/plain", H)
    #     println()
    # end
    # mul!(view(H,:,3:n), givens_first)
    # if debug
    #     str = str * " * G$(indices[n-1])'"
    #     println("Apply from the rhs: $str")
    #     display("text/plain", H)
    #     println()
    # end

    return H_original, H, Q
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