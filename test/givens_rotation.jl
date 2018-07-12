using Test

using IRAM: Givens, Hessenberg

@testset "Givens rotation" begin
    for T in (Float64, ComplexF64)
        H = triu(rand(T, 5,5), -1)
        H[2,1] = 0
        G = Matrix{T}(I, 5, 5)
        c, s = rand(), rand(T)
        G[2:3,2:3] = [c s; -conj(s) c]

        H_new = Hessenberg(copy(H))
        lmul!(Givens(c, s, 2), H_new)

        @test G * H ≈ H_new.H
    end

    for T in (Float64, ComplexF64)
        Q = rand(T, 10, 4)
        G = Matrix{T}(I, 4, 4)
        c, s = rand(), rand(T)
        G[2:3,2:3] = [c s; -conj(s) c]
        Q_new = copy(Q)
        rmul!(Q_new, Givens(c, s, 2))
        @test Q * G' ≈ Q_new
    end
end