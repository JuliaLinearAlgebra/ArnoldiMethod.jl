using BenchmarkTools

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, restarted_arnoldi

function bencharnoldi()
    # criterion = 1e-10
    max_restarts = 50

    A = full(sprand(10000, 10000, 5 / 10000))
    
    @benchmark restarted_arnoldi(B, 10, 50, 1e-10, 50) setup = (B = copy($A))


    # A = rand(Complex128, 50,50)
    
    # # λ = sort!(eigvals(A), by = abs, rev = true)
    # # Base.showarray(STDOUT,[λs, λ],false)
    # @benchmark restarted_arnoldi_2(B, 3, 20) setup = (B = copy($A))

end
