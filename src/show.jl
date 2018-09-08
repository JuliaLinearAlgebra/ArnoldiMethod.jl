import Base: show 

function show(io::IO, h::History)
    if h.converged
        printstyled(io, "Converged", color = :green)
    else
        printstyled(io, "Not converged", color = :red)
    end

    print(io, ": ", h.nconverged, " of ", h.nev, " eigenvalues in ",
          h.mvproducts, " matrix-vector products")

end