using ArnoldiMethod: Arnoldi, iterate_arnoldi!, reinitialize!, partialschur
using SparseArrays

function helloworld(n = 40, m = 10)
    A = spdiagm(0 => [range(0.1, stop=5, length=n-3); 100:102])
    A[n,n-1] = 10.0
    A[n-1,n] = -10.0

    partialschur(A, nev=4, mindim=4, maxdim=8)
end