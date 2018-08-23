"""
Run IRAM until the eigenvectors are approximated to the prescribed tolerance or until 
`maxiter` has been reached.
"""
function partial_schur(A; min = 5, max = 30, nev = min, tol = eps(real(eltype(A))), maxiter = 20, which=LM())
    T = eltype(A)
    n = size(A, 1)
    prods = min 

    # Pre-allocated arnoldi decomp
    arnoldi = Arnoldi{T}(n, max)

    # Start with a random vector
    reinitialize!(arnoldi)

    # Expand
    iterate_arnoldi!(A, arnoldi, 1 : min)

    # min′ is the effective starting point -- may be min - 1 when the last two removed guys
    # are a complex conjugate
    min′ = min

    active = 1
    V_prealloc = Matrix{T}(undef, n, max)

    for restarts = 1 : maxiter
        n = max - active + 1

        iterate_arnoldi!(A, arnoldi, min′ + 1 : max)
        prods += length(min′ + 1 : max)

        @info "Outer iteration start" restarts active

        show_estimates(arnoldi)
        
        # Compute shifts
        λs = compute_shifts(arnoldi.H, active, max, tol)
        sort_vals!(λs, which)

        min′ = implicit_restart!(arnoldi, λs, min, max, active, V_prealloc)
        new_active = detect_convergence!(view(arnoldi.H, active:min′+1, active:min′), tol)
        new_active += active - 1 
        if new_active > active + 1
            # Bring the new locked part of H into upper triangular form
            transform_converged(arnoldi, active, new_active, min′, V_prealloc)
        end

        active = new_active

        @info "Outer iteration end" restarts active

        active > nev && break
    end
    return PartialSchur(view(arnoldi.V,:,1:active - 1), view(arnoldi.H, 1:active - 1, 1:active - 1)), prods
end

function show_estimates(arnoldi::Arnoldi{T}) where {T}
    max = size(arnoldi.H, 2)
    H = arnoldi.H[1:max,1:max]
    h = arnoldi.H[max+1, max]
    Q = Matrix{T}(I, max, max)
    x = zeros(complex(T), max)
    local_schurfact!(H, 1, max, Q)
    for (λ, i) = zip(eigvalues(H), 1 : max)
        fill!(x, zero(complex(T)))
        collect_eigen!(x, H, i)
        @show abs(λ) abs(sum(Q[max, j] * x[j] for j = 1 : i)) * abs(h)
    end
end

"""
Transfrom the converged block into an upper triangular form.
"""
function transform_converged(arnoldi, active, new_active, min′, V_prealloc)
    
    # H = Q R Q'

    # A V = V H
    # A V = V Q R Q'
    # A (V Q) = (V Q) R
    
    # V <- V Q
    # H_right <- Q' H_right
    # H_lock <- Q' H_lock Q
    # H_above <- H_above Q

    Q_large = Matrix{eltype(arnoldi.H)}(I, new_active - 1, new_active - 1)

    local_schurfact!(arnoldi.H, active, new_active - 1, Q_large)
    Q_small = view(Q_large, active : new_active - 1, active : new_active - 1)

    V_locked = view(arnoldi.V, :, active : new_active - 1)
    mul!(view(V_prealloc, :, active : new_active - 1), V_locked, Q_small)
    V_locked .= view(V_prealloc, :, active : new_active - 1)

end

function compute_shifts(H::AbstractMatrix{T}, active, max, tol=100eps(real(T))) where {T}
    n = max - active + 1

    # Compute the eigenvalues of the active part
    Q = Matrix{T}(I, n, n)
    R = H[active:max, active:max]
    local_schurfact!(R, 1, n, Q)
    λs = eigvalues(R)

    return λs

end


function compute_shifts(H::AbstractMatrix{T}, active, max, tol=100eps(T)) where {T<:Real}
    n = max - active + 1

    # Compute the eigenvalues of the active part
    Q = Matrix{T}(I, n, n)
    R = H[active:max, active:max]
    local_schurfact!(R, 1, n, Q)
    λs = eigvalues(R)

    return λs

end
