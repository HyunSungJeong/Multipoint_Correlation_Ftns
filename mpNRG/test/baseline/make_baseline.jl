# Writes the regression baseline of the TNOps refactor with the code at the tag
# `pre-tensor-refactor` (= bb8b197). Run once, from the repository root:
#
#   julia -t 4 --project=mpNRG/test mpNRG/test/baseline/make_baseline.jl
#
# Outputs (committed):
#   baseline.jld2            the quantities of cases.jl, plus the frozen fixtures
#                            of the old internals (fusionmap, kron_op,
#                            truncate_shell, refine, _rotate, FDM backward step)
#   old_output.jld2          a file written by `save_results` (format check)
#   tnops_fixtures.jl        the fusionmap / kron_op / truncate_shell fixtures as
#                            Julia literals, for TNOps/test (no JLD2 there)

using MultipointNRG, LinearAlgebra, JLD2, Random
const M = MultipointNRG
include(joinpath(@__DIR__, "cases.jl"))

const OUT = @__DIR__
commit = strip(read(`git -C $(OUT) rev-parse HEAD`, String))
println("code at ", commit, ", Julia ", VERSION, ", threads ", Threads.nthreads())

# ---------------------------------------------------------------- plain-data forms
blocks_of(A::M.BlockOp) = Dict(skey(q) => copy(B) for (q, B) in A.blocks)
space_of(V::M.SectorSpace) = [(skey(q), d) for (q, d) in zip(V.labels, V.dims)]
function fusion_of(F::M.FusionMap)
    return Dict("loc" => space_of(F.loc), "prev" => space_of(F.prev), "out" => space_of(F.out),
                "pairs" => Dict(skey(q) => [(skey(p.σ), skey(p.q), p.offset, p.dσ, p.dq) for p in prs] for (q, prs) in F.pairs),
                "lookup" => Dict((skey(a), skey(b)) => (skey(c), k) for ((a, b), (c, k)) in F.lookup))
end
op_of(A::M.BlockOp) = Dict("row" => space_of(A.row), "col" => space_of(A.col), "shift" => skey(A.shift), "blocks" => blocks_of(A))

# ---------------------------------------------------------------- baseline quantities
t0 = time()
base = compute_baseline()
println("baseline quantities: ", round(time() - t0; digits = 1), " s")

# ---------------------------------------------------------------- fixtures of the old internals
fx = Dict{String,Any}()
m, p, T = Dict(bl_cases())["trunc"]
run = run_nrg(m, p); f = fdm(run; T = T)
fx["trunc/moduli"] = Tuple(map(x -> x isa M.U1 ? 0 : x.n, m.G.factors))
fx["trunc/n0"] = run.n0
# fusion maps of every shell
for n in 0:p.N
    fx["fusion/$n"] = fusion_of(shell(run, n).fusion)
end
# kron_op: the three terms of H_n at n = 2 and a random-block case
let n = 2
    sh = shell(run, n); prev = shell(run, n - 1); loc = sh.loc; Fu = sh.fusion
    Vloc = M.localspace_sectors(loc); Vk = keptspace(prev); Tt = Float64
    Ik = identity_op(Vk, Tt); Il = identity_op(Vloc, Tt)
    Ek = M._diagop(Vk, Dict(q => prev.E[q][1:nkept(prev, q)] for q in sectors(Vk)), Tt)
    Z = M.parity_op(loc)
    kk = Dict(q => krange(prev, q) for q in sectors(prev.space) if nkept(prev, q) > 0)
    fdag = M.localop(loc, cdag(Mode(n, :up)); T = Tt) * Z
    fprev = restrict(prev.F[1], kk, kk)
    cases = Dict("IlEk" => (Il, Ek), "fdagZ_fprev" => (fdag, fprev),
                 "hloc_Ik" => (M.localop(loc, num(Mode(n, :up)); T = Tt), Ik))
    for (k, (X, Y)) in cases
        fx["kron/$k"] = Dict("X" => op_of(X), "Y" => op_of(Y), "fusion" => fusion_of(Fu), "out" => op_of(kron_op(X, Y, Fu, Fu)))
    end
end
# truncate_shell on every recorded spectrum of every case
tr_list = Any[]
for (name, (mm, pp, TT)) in bl_cases()
    rr = run_nrg(mm, pp)
    for n in -1:pp.N
        sh = shell(rr, n)
        E = Dict(q => copy(sh.E[q]) for q in sectors(sh.space))
        ω = sh.info[:ω]; last = n == pp.N
        nk, rec = M.truncate_shell(E, sh.space, ω; Nkeep = pp.Nkeep, Etrunc = pp.Etrunc, gaptol = pp.gaptol, last = last)
        @assert all(nk[q] == nkept(sh, q) for q in sectors(sh.space))
        push!(tr_list, Dict("case" => name, "n" => n, "moduli" => Tuple(map(x -> x isa M.U1 ? 0 : x.n, mm.G.factors)),
                            "E" => Dict(skey(q) => e for (q, e) in E), "omega" => ω, "Nkeep" => pp.Nkeep,
                            "gaptol" => pp.gaptol, "last" => last,
                            "nk" => Dict(skey(q) => v for (q, v) in nk), "rec" => Dict(string(k) => v for (k, v) in pairs(rec))))
    end
end
fx["truncate"] = tr_list
# refine (odd and even operators, at the first truncated shell and one later),
# _rotate (the site annihilator of the sweep) and the FDM backward step
for n in (run.n0 + 1, run.n0 + 3)
    sh = shell(run, n); prev = shell(run, n - 1)
    fx["A/$n"] = Dict(skey(q) => copy(B) for (q, B) in sh.A)
    fx["nkept/$(n - 1)"] = Dict(skey(q) => nkept(prev, q) for q in sectors(prev.space))
    fx["nkept/$n"] = Dict(skey(q) => nkept(sh, q) for q in sectors(sh.space))
    fx["space/$(n - 1)"] = space_of(prev.space); fx["space/$n"] = space_of(sh.space)
    for (lab, e) in (("d_up", d_(:up)), ("n_dn", n_(:dn)), ("ddag_dn", dd_(:dn)))
        O = M.project(run, e, n - 1)
        fx["refine/$n/$lab"] = Dict("in" => op_of(O), "out" => op_of(M.refine(run, O, n)))
    end
    # _rotate: P = f_n ⊗ 1 on the product space, rotated into the shell basis
    loc = sh.loc; Fu = sh.fusion; Vk = keptspace(prev)
    P = kron_op(M.localop(loc, cop(Mode(n, :up)); T = Float64), identity_op(Vk, Float64), Fu, Fu)
    fx["rotate/$n"] = Dict("P" => op_of(P), "out" => op_of(M._rotate(P, sh.A, sh.A, sh.space, sh.space)))
    @assert norm(todense(M._rotate(P, sh.A, sh.A, sh.space, sh.space)) - todense(sh.F[1])) == 0
    # FDM backward step n → n-1: inputs ρ̄^n_KK, ρ̄^n_DD (diagonal), output ρ̄^{n-1}_KK
    fx["fdm/$n"] = Dict("rhoK_n" => Dict(skey(q) => copy(B) for (q, B) in rhoK(f, n)),
                        "rhoD_n" => Dict(skey(q) => copy(v) for (q, v) in rhoD(f, n)),
                        "rhoK_prev" => Dict(skey(q) => copy(B) for (q, B) in rhoK(f, n - 1)))
end

# ---------------------------------------------------------------- write
jldsave(joinpath(OUT, "baseline.jld2"); baseline = base, fixtures = fx, commit = commit,
        julia = string(VERSION), threads = Threads.nthreads())
println("written baseline.jld2")

# old-format output file (Sec. 5 of the refactor sheet)
let (mm, pp, TT) = Dict(bl_cases())["untrunc"]
    rr = run_nrg(mm, pp); ff = fdm(rr; T = TT)
    ps = psfs(rr, ff, (d_(:up), dd_(:up)); grid = BL_GRID)
    ps[[0]] = psf(rr, ff, (d_(:up), dd_(:up)); exact = true)
    save_results(joinpath(OUT, "old_output.jld2"), rr, ff, ps; thermal = Dict("n_up" => thermal_average(rr, ff, n_(:up))))
    println("written old_output.jld2")
end

# ---------------------------------------------------------------- TNOps fixtures as Julia literals
function lit(x)
    x isa AbstractDict && return "Dict(" * join(["$(lit(k)) => $(lit(v))" for (k, v) in sort(collect(x); by = first)], ", ") * ")"
    x isa Tuple && return "(" * join(lit.(x), ", ") * (length(x) == 1 ? ",)" : ")")
    x isa AbstractMatrix && return "reshape(Float64[" * join(repr.(vec(x)), ", ") * "], $(size(x, 1)), $(size(x, 2)))"
    x isa AbstractVector && return "Any[" * join(lit.(x), ", ") * "]"
    x isa Float64 && return repr(x)
    return repr(x)
end
open(joinpath(OUT, "tnops_fixtures.jl"), "w") do io
    println(io, "# Frozen output of the pre-refactor fusionmap / kron_op / truncate_shell")
    println(io, "# (mpNRG at $(commit), tag pre-tensor-refactor), written by")
    println(io, "# mpNRG/test/baseline/make_baseline.jl. Labels are plain tuples; `moduli`")
    println(io, "# gives the factor moduli (0 for U(1)), the parity factor last.")
    println(io, "const FX_MODULI = ", lit(fx["trunc/moduli"]))
    println(io, "const FX_FUSION = ", lit(Dict(n => fx["fusion/$n"] for n in 0:p.N)))
    println(io, "const FX_KRON = ", lit(Dict(k => fx["kron/$k"] for k in ("IlEk", "fdagZ_fprev", "hloc_Ik"))))
    println(io, "const FX_TRUNCATE = ", lit(fx["truncate"]))
end
println("written tnops_fixtures.jl")
