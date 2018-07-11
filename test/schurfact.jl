using Base.Test

using IRAM: mul!, Givens, Hessenberg, ListOfRotations, qr!, implicit_restart!, initialize, iterate_arnoldi!, restarted_arnoldi, eigvalues, schurfact!

@testset "Schur factorization" begin

    # 2-by-2 matrix with distinct eigenvalues while H[2,1] != 0
    H = [1.0 2.0; 3.0 4.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with distinct eigenvalues while H[2,1] = 0
    H = [1.0 2.0; 0.0 4.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8


    # 2-by-2 matrix with conjugate eigenvalues
    H = [1.0 4.0; -5.0 3.0]
    H_copy = copy(H)
    Q = eye(Float64, 2)
    schurfact!(H, Q, 1, 2)

    @test vecnorm(Q' * H_copy * Q - H) < 1e-8
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H), by = abs, rev = true)
    @test sort!(eigvalues(H), by = abs, rev = true) ≈ sort!(eigvals(H_copy), by = abs, rev = true)
    @test H[2,1] < 1e-8
    

    # # Transforming a 5 by 5 block in different positions of the matrix H into upper triangular form
    # for i = 1 : 6
    #     H = triu(rand(10,10), -1)
    #     λs = sort!(eigvals(H), by=abs, rev=true)
    #     H_copy = copy(H)
    #     Q = eye(10)
    #     schurfact!(H, Q, i, i+4)
    #     λs_transformed = eigvals(H[i:i+4,i:i+4])

    #     for j = 0 : 3
    #         t = H[i+j,i+j] + H[i+j+1,i+j+1]
    #         d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

    #         # Test if sbdiagonal is small. If not, check if conjugate eigenvalues.
    #         @test H[i+j+1,i+j] < 1e-8 || t*t/4 - d < 0
    #         @show H[i+j+1,i+j]
    #         @show (t) * (t) / 4 - (d)
    #         @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
    #     end

    #     display(H)
    #     @test vecnorm(Q[i:i+4,i:i+4]*H[i:i+4,i:i+4]*Q[i:i+4,i:i+4]' - H_copy[i:i+4,i:i+4]) < 1e-8
    #     @test λs ≈ sort!(eigvals(H), by=abs, rev=true)

    # end
    
    # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    for i = 0 : 4
        H = triu(rand(10,10), -1)
        λs = sort!(eigvals(H), by=abs, rev=true)
        H_copy = copy(H)
        Q = eye(10)

        # Test that the procedure has converged
        @test schurfact!(H, Q, 1+i, 10-i, maxiter=10)

        for j = 1 : 9 - 2*i
            t = H[i+j,i+j] + H[i+j+1,i+j+1]
            d = H[i+j,i+j]*H[i+j+1,i+j+1] - H[i+j+1,i+j]*H[i+j,i+j+1]

            # Test if subdiagonal is small. If not, check if conjugate eigenvalues.
            @test H[i+j+1,i+j] < 1e-8 || t*t/4 - d < 0
            @show H[i+j+1,i+j]
            @show (t) * (t) / 4 - (d)
            @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
        end
        @show 1+i : 10-i
        display(H)
        display(H_copy)

        # Test that the elements below the subdiagonal are 0
        for j = 1:10
            for i = j+2:10
                @test H[i,j] < 1e-8
            end
        end

        # Test that the partial Schur decomposition relation holds
        @test vecnorm(Q*H*Q' - H_copy) < 1e-8
        
        # Test that the eigenvalues of H are the same before and after transformation
        @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
        
        @show sort!(eigvals(H), by=abs, rev=true)
    end


    # COMPLEX ARITHMETIC

    #     # Transforming a 1+i by 10-i block of the matrix H into upper triangular form
    # for i = 0 : 4
    #     H = triu(rand(Complex128, 10,10), -1)
    #     λs = sort!(eigvals(H), by=abs, rev=true)
    #     H_copy = copy(H)
    #     Q = eye(Complex128, 10)

    #     # Test that the procedure has converged
    #     schurfact!(H, Q, 1+i, 10-i)

    #     for j = 1 : 9 - 2*i  
    #         # Test if subdiagonal is small. 
    #         @test abs(H[i+j+1,i+j]) < 1e-8
    #         @show H[i+j+1,i+j]
    #         @show eigvals(H[i+j:i+j+1,i+j:i+j+1])
    #     end
    #     @show 1+i : 10-i
    #     display(H)

    #     # Test that the elements below the subdiagonal are 0
    #     for j = 1:10
    #         for i = j+2:10
    #             @test abs(H[i,j]) < 1e-8
    #         end
    #     end

    #     # Test that the partial Schur decomposition relation holds
    #     @test vecnorm(Q*H*Q' - H_copy) < 1e-8
        
    #     # Test that the eigenvalues of H are the same before and after transformation
    #     @test λs ≈ sort!(eigvals(H), by=abs, rev=true)
        
    #     @show sort!(eigvals(H), by=abs, rev=true)
    # end

    # H2 = [ 31.3265       -4.09316       1.95146      -0.770332     -0.220247      0.451145;
    # 0.133979     30.4833       -2.25368       0.70086      -0.456425      0.255826;   
    # 0.0           0.255179     29.8498       -1.13323       0.21986      -0.157696;      
    # 3.38813e-21   8.67362e-19   0.410006     29.1263       -0.239704      0.0348652;     
    # 2.97702e-22   0.0          -1.73472e-18   0.447611     29.0287       -0.216972;      
    # 0.0           5.36667e-22   0.0           1.73472e-18   0.549446     29.2115]

#     H2 = [31.1882       -3.43556       2.12924      -1.20143       0.373494     -0.444468      0.0263278     0.266044      0.025016     -0.0999655     0.149156      0.124794     -0.284588      0.762295     -0.200721  -0.130007   -0.452678;
#     0.119808     30.5719       -2.21098       0.872341     -0.133792      0.00730276   -0.123599      0.0457168     0.0956473     0.0231966     0.532139      0.295859      0.0180042    -0.0543141    -0.35427      0.188485   -0.118822;
#     0.0           0.244926     29.9444       -1.22962       0.165232     -0.254661     -0.228227     -0.0165602    -0.429184      0.0580833     0.0186242    -0.0774268     0.53423       0.662698     -0.114703     0.0622034   0.468389;
#     0.0           0.0           0.334257     29.3668       -0.449183     -0.190511     -0.00199882   -0.396618     -0.658838      0.182753      0.0103548    -0.301003     -0.118031      0.244931      0.251561    -0.105294   -0.24977;
#    -2.39852e-22   6.77626e-21  -8.67362e-19   0.37768      28.9189       -0.0650395    -0.00210953   -0.358496     -0.338258      0.106935     -0.248551     -0.135179     -0.347101      0.31276      -0.0542638   -1.00895     0.391492;
#     0.0          -3.02623e-23   0.0          -1.73472e-18   0.752025     29.3657       -0.0618752    -0.54674      -0.257342     -0.0566419     0.162619      0.529671      0.310831      0.427922     -0.0732926   0.160082    0.483471;
#     0.0           0.0          -5.13387e-22   0.0           0.0           0.274748     29.2475       -0.168643     -0.171152     -0.259148      0.270957     -0.10349      -0.0654297    -0.211849     -0.00814101  -0.508399   -0.361215;
#     1.20499e-26   0.0           0.0          -5.48903e-22   0.0           0.0           0.828307     29.222        -0.344645      0.0129914    -0.0593891    -0.189996     -0.314818      0.252799      0.161487     0.0709056  -0.482193;
#    -3.25663e-28  -2.59391e-27   0.0           0.0          -6.76723e-22   0.0          -1.73472e-18   0.337935     29.1961       -0.0369448     0.495392      0.47259       0.0200621    -0.246639      0.219262     0.0622093   0.131451;
#     1.5879e-31    1.88458e-30   1.29644e-28   0.0           0.0          -5.19566e-24  -8.47033e-22  -2.71051e-20   0.00773439   29.5885       -0.402548      0.254015     -0.282774     -0.148623      0.129625    -0.0111986  -0.160152;
#    -8.84379e-34  -1.18083e-32  -7.16064e-31   1.68842e-29   0.0           0.0           2.02884e-24   0.0           6.77626e-21   0.102539     29.7074       -0.455472      0.245738     -0.35775       0.00629876   0.365287    0.116134;
#     6.29608e-36   8.88233e-35   5.02062e-33  -1.31072e-31  -1.07176e-30   0.0           0.0          -3.31328e-25   0.0           0.0           0.128563     29.2412        0.308644      0.0429495    -0.307015    0.241037    0.202759;
#    -1.14114e-38  -1.64821e-37  -8.9583e-36    2.47219e-34   1.15194e-32   4.78169e-31   0.0           0.0           3.40346e-26   0.0          -2.71051e-20   0.0366382    29.3337       -0.552039     -0.154715     -0.0233062  -0.189498;
#     2.96916e-41   3.87165e-40   2.05968e-38  -5.86508e-37  -1.13134e-34  -7.65032e-33  -5.46215e-32   0.0           0.0          -3.30625e-25   0.0          -1.35525e-20   0.0525632    29.8763        0.606362     -0.174043    0.531885;
#    -5.58925e-43  -5.81106e-43  -3.35413e-41   9.7855e-40    1.18885e-35   9.98755e-34   1.05995e-34   1.61539e-33   0.0           0.0          -4.13374e-27   0.0          -1.35525e-20   0.0342579    29.4822       -0.157873   -0.345075;
#     2.79592e-45  -1.53068e-46   9.13434e-45  -2.71018e-43  -6.37929e-38  -5.37976e-36  -3.22915e-38  -4.11951e-37  -4.38961e-35   0.0           0.0           1.70063e-22   0.0           0.0           0.00600456  29.4707      0.0432506;
#    -2.43434e-45   2.88088e-46  -3.36208e-47   8.00355e-46   5.57568e-38   4.7032e-36    1.01743e-40   1.13949e-39   1.60616e-37   5.94074e-34   0.0           0.0           9.15214e-27  -3.30872e-24   0.0         0.0652714  29.4785]
#     @show eigvals(H2)
#     H_copy2 = copy(H2)
#     Q2 = eye(17)
#     schurfact!(H2, Q2, 1, 17)
#     display(H2)
#     @test vecnorm(Q2*H2*Q2' - H_copy2) < 1e-8

    # @show eigvals(H2)
    # H_copy2 = copy(H2)
    # Q2 = eye(17)
    # schurfact!(H2, Q2, 1, 17)
    # display(H2)
    # @test vecnorm(Q2*H2*Q2' - H_copy2) < 1e-8
    
    
end
