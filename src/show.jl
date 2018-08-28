import Base: show 

function show(io::IO, h::History)
    if h.converged
        print(io, "Converged after ")
    else
        print(io, "Not converged after ")
    end
    print(io, h.mvproducts, " matrix-vector products")
end