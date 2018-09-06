using LinearAlgebra, SparseArrays
using Test
using ArnoldiMethod: Rotation2, Arnoldi, iterate_arnoldi!, reinitialize!, local_schurfact!

"""
    rotate_to_eₖ(y) -> z, τ

Returns z, τ such that (I - 2zz') y = eₖ ‖y‖ τ and |τ| = 1.

Based on Mezzadri, Francesco. "How to generate random matrices from the 
classical compact groups." arXiv preprint math-ph/0609050 (2006).
"""
function rotate_to_eₖ(y, k::Integer = 1)
    @inbounds begin
        τ = y[k] / abs(y[k])
        z = copy(y)
        z[k] += τ * norm(y)
        z ./= norm(z)
        return z, -τ
    end
end

function testetst(n = 10)
    @testset "reflection $i" for i = 1 : 10
        b = rand(ComplexF64, n)
        z, τ = rotate_to_eₖ(b, 1)
        e₁ = zeros(n)
        e₁[1] = 1.0
        @test abs(τ) ≈ 1
        @test (b - 2 * dot(z, b) * z) ≈ e₁ * τ * norm(b)
    end
end


function example(n = 10, ::Type{T} = ComplexF64) where {T<:Number}
    # Random Hessenberg matrix
    H = triu(rand(T, n, n), -1)
    H₀ = copy(H)

    # Find an eigenvalue + vector
    λs, xs = eigen(H)
    λ, x = λs[2], xs[:, 2]

    # Find a reflector that maps x -> e₁
    z, τ = rotate_to_eₖ(x, 1)
    W₁ = I - 2 * (z * z')

    # Apply it to the Hessenberg matrix s.t. H[1,1] = λ and H[2,1] = 0
    H = W₁ * H * W₁

    Q = Matrix{T}(I, n, n)

    # Restore Hessenberg structure (row by row, not column by column!)
    # This order is very important, cause this is the way to retain the
    # Hessenberg structure.
    for i = n-1:-1:3
        h = conj(H[i+1, 2:i]) # ugliness
        z₂, τ₂ = rotate_to_eₖ(h, i-1)
        W₂ = Matrix{ComplexF64}(I, n, n)
        W₂[2:i,2:i] .= I - 2 * (z₂ * z₂')
        H = W₂ * H * W₂
        Q *= W₂
    end

    # In the end we have Q' * W' * H₀ * W * Q = H
    # So with the Arnoldi relation
    # A*V = V*H₀ + h*v*e'
    # AV(WQ) = V(WQ)(WQ)'H₀(WQ) + h v e' (WQ)
    # AV(WQ) = V(WQ)H + h v (e' + w') Q
    # AV(WQ) = V(WQ)H + h v e' + h v w' Q
    # ‖h v w' Q‖ = |h| ‖w‖ ≤ √2 |yₖ|
    # drop the last term: AV(WQ) = V(WQ) + h v e'.

    return H₀, H, W₁, Q
end

function sometesting(n)
    H = rand(ComplexF64, n + 1, n)
    h = conj(H[end, :])
    z, τ = rotate_to_eₖ(h, n)
    Q = I - 2 * (z * z')
    H *= Q
    H[1:n, 1:n] .= Q * H[1:n,1:n]
    return H
end

function H_with_conjugate(n)
    while true
        H = triu(randn(n, n), -1)
        λs, xs = eigen(H)
        i = findfirst(!iszero ∘ imag, λs)
        if i !== nothing
            return H, λs[i], xs[:, i]
        end
    end
end

function more_testing(n = 10)
    H, λ, x = H_with_conjugate(n)
    X = [real(x) imag(x)]
    Q, R = qr(X)

    Q' * H * Q, Q
end

function random_unitary_hessenberg_matrix(n::Int = 10, ::Type{T} = Float64) where {T}
    H = Matrix{T}(I, n, n)

    for i = n-1:-1:1
        θ = 2π * rand()
        lmul!(Rotation2(cos(θ), sin(θ), i), H, i, n)
    end

    H
end

function example_arnoldi_relation(n = 40, m = 10)
    # A = spdiagm(0 => [range(0.1, stop=5, length=n-3); 100:102])
    # A[n,n-1] = 1.0
    # A[n-1,n] = -1.0

    A = rand(n, n)

    arn = Arnoldi{Float64}(n, m)
    reinitialize!(arn)
    iterate_arnoldi!(A, arn, 1:m)
    V = arn.V
    H = arn.H
    
    # Random unitary matrix
    Q = random_unitary_hessenberg_matrix(m)

    # Destroy the Schur decomp
    Hnext = H * Q
    Hnext[1:m,1:m] .= Q' * Hnext[1:m,1:m]

    # Restore last row of H
    for i = m:-1:2
        z, = rotate_to_eₖ(conj(Hnext[i+1,1:i]), i)
        W = Matrix{ComplexF64}(I, m, m)
        W[1:i,1:i] .= I - 2 * (z * z')
        Hnext .= Hnext * W
        Hnext[1:m,1:m] .= W' * Hnext[1:m,1:m]
        Q *= W
    end

    W = [V[:,1:m] * Q V[:,m+1]]

    @show opnorm(W' * W - I) opnorm(V' * V - I)
    @show opnorm(A * W[:,1:m] - W * Hnext) opnorm(A * V[:,1:m] - V * H)
end