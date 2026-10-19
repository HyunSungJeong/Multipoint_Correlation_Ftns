# Multi-tensor contraction with a caller-given order (no order optimization).

"""
    ncon(tensors, labels; order = sort(positive labels)) -> BlockTensor

Contract a network in the ncon convention: `labels[t][i]` labels leg `i` of
tensor `t`; a positive label appears exactly twice and is contracted, a
negative label `-j` marks open leg `j` of the result. The positive labels are
contracted in the given `order`: at each step the two tensors carrying the
label are contracted over all labels they share (or, if both legs belong to
one tensor, that pair is traced). Disconnected parts are joined by outer
products at the end; the open legs are finally permuted to `-1, -2, …`.
"""
function ncon(tensors::AbstractVector, labels::AbstractVector; order = nothing)
    length(tensors) == length(labels) || throw(ArgumentError("one label list per tensor"))
    ts = Any[t for t in tensors]; ls = [collect(Int, l) for l in labels]
    for (t, l) in zip(ts, ls)
        ndims(t) == length(l) || throw(ArgumentError("a tensor of rank $(ndims(t)) has $(length(l)) labels"))
    end
    allp = sort!(unique(x for l in ls for x in l if x > 0))
    for x in allp
        count(==(x), reduce(vcat, ls)) == 2 || throw(ArgumentError("label $x must appear exactly twice"))
    end
    ord = order === nothing ? allp : collect(Int, order)
    sort(ord) == allp || throw(ArgumentError("order must list every positive label once"))
    done = Set{Int}()
    for x in ord
        x in done && continue
        holders = [t for t in eachindex(ts) if x in ls[t]]
        if length(holders) == 1
            t = holders[1]
            i, j = findall(==(x), ls[t])
            ts[t] = tr(ts[t], i, j)
            ls[t] = [y for (k, y) in enumerate(ls[t]) if k != i && k != j]
            push!(done, x)
        else
            a, b = holders
            common = [y for y in ls[a] if y > 0 && y in ls[b]]
            ia = Tuple(findfirst(==(y), ls[a]) for y in common)
            ib = Tuple(findfirst(==(y), ls[b]) for y in common)
            C = contract(ts[a], ia, ts[b], ib)
            newl = vcat([y for y in ls[a] if !(y in common)], [y for y in ls[b] if !(y in common)])
            union!(done, common)
            keep = [t for t in eachindex(ts) if t != a && t != b]
            ts = Any[ts[keep]..., C]; ls = [ls[keep]..., newl]
        end
    end
    while length(ts) > 1
        C = outer(ts[1], ts[2])
        ts = Any[C, ts[3:end]...]; ls = [vcat(ls[1], ls[2]), ls[3:end]...]
    end
    R = ts[1]; l = ls[1]
    all(<(0), l) || throw(ArgumentError("uncontracted positive labels remain"))
    p = Tuple(findfirst(==(-j), l) for j in 1:length(l))
    any(isnothing, p) && throw(ArgumentError("open labels must be -1, -2, …, -$(length(l))"))
    return length(l) ≤ 1 ? R : permute(R, p)
end
