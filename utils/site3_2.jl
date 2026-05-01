using ITensors
using LinearAlgebra  # <-- add this
import ITensors: SiteType, OpName, op
import ITensors.SiteTypes: @SiteType_str, @OpName_str
import ITensors.SiteTypes: siteinds

# using ITensors.SiteTypes  # this brings in the SiteType"..." and OpName"..." macros

# --- Spin 3/2 SiteType Definition ------------------------------------
ITensors.space(::SiteType"S=3/2") = 4

function ITensors.op(::OpName"Sz", ::SiteType"S=3/2")
    return Diagonal([3/2, 1/2, -1/2, -3/2])
end

function ITensors.op(::OpName"S+", ::SiteType"S=3/2")
    M = zeros(4, 4)
    M[1, 2] = √3
    M[2, 3] = 2
    M[3, 4] = √3
    return M
end

function ITensors.op(::OpName"S-", ::SiteType"S=3/2")
    M = zeros(4, 4)
    M[2, 1] = √3
    M[3, 2] = 2
    M[4, 3] = √3
    return M
end

function ITensors.op(::OpName"Sx", ::SiteType"S=3/2")
    return 0.5 * (op("S+", SiteType("S=3/2")) + op("S-", SiteType("S=3/2")))
end

function ITensors.op(::OpName"Sy", ::SiteType"S=3/2")
    return 0.5im * (op("S-", SiteType("S=3/2")) - op("S+", SiteType("S=3/2")))
end

function ITensors.op(::OpName"Id", ::SiteType"S=3/2")
    return Matrix(I, 4, 4)
end

# Optional exponentials
function ITensors.op(::OpName"exp(i*pi*Sx)", ::SiteType"S=3/2")
    return exp(im * π * op("Sx", SiteType("S=3/2")))
end

function ITensors.op(::OpName"exp(i*pi*Sy)", ::SiteType"S=3/2")
    return exp(im * π * op("Sy", SiteType("S=3/2")))
end

function ITensors.op(::OpName"exp(i*pi*Sz)", ::SiteType"S=3/2")
    return exp(im * π * op("Sz", SiteType("S=3/2")))
end
# ---------------------------------------------------------------------

# sites = siteinds("S=3/2", 4)
# display(op("Sz", sites[1]))