using Test
using ArnoldiMethod: sylv
using StaticArrays
using LinearAlgebra

@testset "Tiny sylvester equation $T, $s" for T in (Float64, ComplexF64), s = ((2,2),(2,1),(1,2))
    p, q = s
    A = @SMatrix rand(T, p, p)
    B = @SMatrix rand(T, q, q)
    C = @SMatrix rand(T, p, q)

    X, singular = sylv(A, B, C)

    @test A * X - X * B ≈ C
    @test !singular
end

@testset "Singular sylvester $T" for T in (Float64, ComplexF64)
    let
        A = @SMatrix T[1 2; 0 1]
        B = @SMatrix T[1 3; 0 1]
        C = @SMatrix rand(T, 2, 2)
        X, singular = sylv(A, B, C)
        @test singular
    end
    let
        A = @SMatrix T[1]
        B = @SMatrix T[1 3; 0 1]
        C = @SMatrix rand(T, 1, 2)
        X, singular = sylv(A, B, C)
        @test singular
    end
    let
        A = @SMatrix T[1 2; 0 1]
        B = @SMatrix T[1]
        C = @SMatrix rand(T, 2, 1)
        X, singular = sylv(A, B, C)
        @test singular
    end
end