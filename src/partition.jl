
"""
    partition!(predicate, v::AbstractVector, r::AbstractRange) → Union{Nothing,Int}

Calling `k = partition!(f, v)` reorders `v` such that `all(f, v[k:end])` and
`all(x -> !f(x), v[1:k-1]` are true. If there is no element in `v` for which the
predicate is false, then `k` is `nothing`.
"""
function partition!(predicate, v::AbstractVector, range::AbstractRange = 1 : length(v))
    @inbounds begin
        isempty(range) && return nothing
        
        idx = range[1]
        to = range[end]

        while idx ≤ to && predicate(v[idx])
            idx += 1
        end

        idx > to && return nothing

        for j = idx+1:to
            if predicate(v[j])
                v[idx], v[j] = v[j], v[idx]
                idx += 1
            end
        end
        return idx
    end
end
