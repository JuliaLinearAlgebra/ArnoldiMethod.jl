using Base.Test
using IRAM: qr!, Hessenberg, mul!

@testset "QR on Hessenberg matrix" begin
    # Random Hessenberg matrix of size 11 x 10.
    H = triu(rand(11, 10), -1)
    
    H_triu = Hessenberg(copy(H))

    rotations = qr!(H_triu)
    Q = mul!(eye(11), rotations)

    @test H ≈ Q * H_triu.H
    # @test istriu(H)
end