using Test
using ArnoldiMethod: reflector!, restore_arnoldi!, Reflector, Arnoldi, reinitialize!, iterate_arnoldi!
using LinearAlgebra
using Random

@testset "Reflector $T" for T in (Float64, ComplexF64)
    n = 20
    x = rand(T, n)
    z = copy(x)
    τ = reflector!(z, n)
    z[n] = 1
    y = x .- τ * dot(z, x) .* z

    @inferred reflector!(copy(x), n)
    @test norm(y[1:n-1]) ≤ 10eps()
    @test abs(real(y[n])) ≈ norm(x) 
    @test abs(imag(y[n])) ≤ eps()
    @test 1 ≤ real(τ) ≤ 2
    @test abs(τ - 1) ≤ 1
end

@testset "Trivial reflector $T" for T in (Float64, ComplexF64)
    x = [T(0), T(0), T(5)]
    z = copy(x)
    τ = reflector!(z, 3)
    @test iszero(τ)
end

@testset "rmul! $T" for T in (Float64, ComplexF64)
    A = rand(T, 4, 4)
    B = copy(A)
    
    # Implicit representation
    G = Reflector{T}(4)
    rand!(G.vec)
    τ = reflector!(G, 4)

    # Matrix representation
    z = [G.vec[1:3]; 1]
    H = I - τ * z * z'

    # Apply
    rmul!(A, G, 1, 4)

    @test A ≈ B * H'
end

@testset "lmul! $T" for T in (Float64, ComplexF64)
    A = rand(T, 4, 4)
    B = copy(A)
    
    # Implicit representation
    G = Reflector{T}(4)
    rand!(G.vec)
    τ = reflector!(G, 4)

    # Matrix representation
    z = [G.vec[1:3]; 1]
    H = I - τ * z * z'

    # Apply
    lmul!(G, A, 1, 4)

    @test A ≈ H * B
end

@testset "Restore Hessenberg matrix $T" for T in (Float64, ComplexF64)
    n, k = 10, 6
    Q = qr(randn(T, k, k)).Q * Matrix(1.0I, k, k)

    A = rand(T, n, n)
    arn = Arnoldi{T}(n, k)
    reinitialize!(arn)
    iterate_arnoldi!(A, arn, 1:k)

    # Do an orthogonal change of basis.

    H = copy(arn.H)
    H[1:k,1:k] .= Q' * H[1:k,1:k] * Q

    # Restore the relation
    restore_arnoldi!(H, 1, k, Q, Reflector{T}(k))

    # Only now do the change of basis in V with accumulated Q.
    W = [arn.V[:, 1:k] * Q arn.V[:, k+1]]
    @test norm(A * W[:, 1:k] - W[:, 1:k+1] * H) < 1e-14
end