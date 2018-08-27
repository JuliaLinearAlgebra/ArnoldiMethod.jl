
"""
    partition!(predicate, v::AbstractVector) -> Union{Nothing,Int}

Calling `k = partition!(f, v)` reorders `v` such that `all(f, v[k:end])` and
`all(x -> !f(x), v[1:k-1]` are true. If there is no element in `v` for which the
predicate is false, then `k` is `nothing`.
"""
function partition!(predicate, v::AbstractVector)
    first = findfirst(x -> !predicate(x), v)
    first === nothing && return nothing
    @inbounds for j = first+1:length(v)
        if predicate(v[j])
            v[first], v[j] = v[j], v[first]
            first += 1
        end
    end
    first
end
