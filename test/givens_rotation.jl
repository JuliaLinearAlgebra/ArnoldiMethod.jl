using Base.Test

using IRAM: mul!, Givens, Hessenberg

@testset "Givens rotation" begin
    for T in (Float64, Complex128)
        H = triu(rand(T, 5,5), -1)
        H[2,1] = 0
        G = eye(T, 5)
        c, s = rand(), rand(T)
        G[2:3,2:3] = [c s; -conj(s) c]

        H_new = Hessenberg(copy(H))
        mul!(Givens(c, s, 2), H_new)

        @test G * H ≈ H_new.H
    end

    for T in (Float64, Complex128)
        Q = rand(T, 10, 4)
        G = eye(T, 4, 4)
        c, s = rand(), rand(T)
        G[2:3,2:3] = [c s; -conj(s) c]
        Q_new = copy(Q)
        mul!(Q_new, Givens(c, s, 2))
        @test Q * G' ≈ Q_new
    end
end