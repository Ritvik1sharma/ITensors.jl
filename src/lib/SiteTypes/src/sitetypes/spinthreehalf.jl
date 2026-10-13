# "S=3/2": spin 3/2, dimension 4, basis m = 3/2, 1/2, -1/2, -3/2 (same order as "S=1").
# Moved here from utils/site3_2.jl (same operators and values) so that `using ITensors`
# provides it.

space(::SiteType"S=3/2") = 4

op(::OpName"Sz", ::SiteType"S=3/2") = [
    3 / 2 0.0 0.0 0.0
    0.0 1 / 2 0.0 0.0
    0.0 0.0 -1 / 2 0.0
    0.0 0.0 0.0 -3 / 2
]

op(::OpName"S+", ::SiteType"S=3/2") = [
    0.0 √3 0.0 0.0
    0.0 0.0 2.0 0.0
    0.0 0.0 0.0 √3
    0.0 0.0 0.0 0.0
]

op(::OpName"S-", ::SiteType"S=3/2") = [
    0.0 0.0 0.0 0.0
    √3 0.0 0.0 0.0
    0.0 2.0 0.0 0.0
    0.0 0.0 √3 0.0
]

op(::OpName"Sx", t::SiteType"S=3/2") = 0.5 * (op(OpName("S+"), t) + op(OpName("S-"), t))

op(::OpName"Sy", t::SiteType"S=3/2") = 0.5im * (op(OpName("S-"), t) - op(OpName("S+"), t))

op(::OpName"Id", ::SiteType"S=3/2") = [
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
]

op(::OpName"exp(i*pi*Sx)", t::SiteType"S=3/2") = exp(im * π * op(OpName("Sx"), t))

op(::OpName"exp(i*pi*Sy)", t::SiteType"S=3/2") = exp(im * π * op(OpName("Sy"), t))

op(::OpName"exp(i*pi*Sz)", t::SiteType"S=3/2") = exp(im * π * op(OpName("Sz"), t))
