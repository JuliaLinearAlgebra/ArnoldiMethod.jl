using Base.Test

using IRAM: mul!, Givens, Hessenberg

@testset "Givens rotation" begin
    H = triu(rand(5,5), -1)
    H[2,1] = 0
    G = eye(5)
    c, s = rand(), rand()
    G[2:3,2:3] = [c s; -s c]

    H_new = Hessenberg(copy(H))
    mul!(Givens(c, s, 2), H_new)

    @test G * H ≈ H_new.H
end