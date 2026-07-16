# General spin-S site types for ITensors.
#
# ITensors ships only "S=1/2" and "S=1". This util adds arbitrary spin: it
# provides a general extractor for the spin-S operator matrices (Sz, S±, Sx, Sy,
# Id) and a registrar that defines the `space` + `op` methods for ANY "S=..."
# tag (e.g. "S=3/2", "S=2", "S=9/2"). Once Sx/Sy exist, ITensors' generic
# op-expression parser handles exp(i*pi*Sy) etc. automatically.
#
# Usage:
#   include(".../ITensors.jl/utils/general_spin.jl")
#   register_spin_sitetype!("S=2")          # or use a pre-registered one
#   s = siteinds("S=2", N)                  # dim-5 sites
#   op("Sy", s[1]); op("exp(i*pi*Sy)", s[1])
#
# Common spins are pre-registered at include time (see bottom).

using ITensors
using LinearAlgebra
import ITensors: SiteType, OpName, op, space

"""
    spin_operator_matrices(d::Integer) -> NamedTuple

Dense spin-S operator matrices for a `d`-dimensional site (S = (d-1)/2), in the
basis ordered m = S, S-1, …, -S (index 1 = highest m). Returns
`(; Sz, Sp, Sm, Sx, Sy, Id)` as `ComplexF64` matrices. S± use the standard
ladder coefficients √(S(S+1) − m(m±1)); Sx = (S⁺+S⁻)/2, Sy = (S⁻−S⁺)/(2i).
"""
function spin_operator_matrices(d::Integer)
    d >= 1 || error("site dimension must be >= 1, got $d")
    S  = (d - 1) / 2
    ms = [S - (k - 1) for k in 1:d]
    Sz = Matrix{ComplexF64}(Diagonal(ms))
    Sp = zeros(ComplexF64, d, d); Sm = zeros(ComplexF64, d, d)
    for k in 1:d-1                                    # S⁺ raises m: ⟨k|S⁺|k+1⟩ (k+1 is the lower m)
        m = ms[k+1]
        c = sqrt(S*(S+1) - m*(m+1))
        Sp[k, k+1] = c; Sm[k+1, k] = c
    end
    Sx = 0.5   * (Sp + Sm)
    Sy = 0.5im * (Sm - Sp)
    return (; Sz, Sp, Sm, Sx, Sy, Id = Matrix{ComplexF64}(I, d, d))
end

"""
    spin_dim_from_tag(tag::AbstractString) -> Int

Parse an `"S=..."` tag (e.g. "S=3/2", "S=2", "S=9/2") into the site dimension
d = 2S + 1.
"""
function spin_dim_from_tag(tag::AbstractString)
    startswith(tag, "S=") || error("expected an \"S=...\" tag, got \"$tag\"")
    frac = tag[3:end]
    S = if occursin('/', frac)
        n, dd = split(frac, '/')
        parse(Float64, n) / parse(Float64, dd)
    else
        parse(Float64, frac)
    end
    d = round(Int, 2S + 1)
    d >= 1 || error("tag \"$tag\" gave nonpositive dimension $d")
    return d
end

"""
    register_spin_sitetype!(tag::AbstractString) -> Int

Define `ITensors.space` and the `Sz/S+/S-/Sx/Sy/Id` `op` methods for the site
type named `tag` (parsed to dimension d = 2S+1), returning d. Skips the native
ITensors spins ("S=1/2", "S=½", "S=1"). Idempotent (redefines on re-call).

NOTE: the SiteType parameter is a `Tag`-encoded type, NOT `Symbol(tag)`, so the
type is obtained via `typeof(SiteType(tag))` (constructing `SiteType{Symbol(tag)}`
by hand does NOT match what `siteinds` dispatches on).
"""
function register_spin_sitetype!(tag::AbstractString)
    (tag in ("S=1/2", "S=½", "S=1")) && return spin_dim_from_tag(tag)   # keep ITensors' native defs
    d  = spin_dim_from_tag(tag)
    M  = spin_operator_matrices(d)
    ST = typeof(SiteType(tag))
    @eval ITensors.space(::$ST) = $d
    @eval ITensors.op(::OpName"Sz", ::$ST) = copy($(M.Sz))
    @eval ITensors.op(::OpName"S+", ::$ST) = copy($(M.Sp))
    @eval ITensors.op(::OpName"S-", ::$ST) = copy($(M.Sm))
    @eval ITensors.op(::OpName"Sx", ::$ST) = copy($(M.Sx))
    @eval ITensors.op(::OpName"Sy", ::$ST) = copy($(M.Sy))
    @eval ITensors.op(::OpName"Id", ::$ST) = copy($(M.Id))
    return d
end

# Pre-register common higher spins (built-ins S=1/2, S=1 are left native).
for _t in ("S=3/2", "S=2", "S=5/2", "S=3", "S=7/2", "S=4", "S=9/2", "S=5")
    register_spin_sitetype!(_t)
end
