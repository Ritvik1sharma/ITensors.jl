# "S=3/2" is now built into ITensors: src/lib/SiteTypes/src/sitetypes/spinthreehalf.jl
# (same space and operators that used to be defined here). This file is kept so scripts
# that include it still load; it only brings the same names into scope as before.
using ITensors
using LinearAlgebra
import ITensors: SiteType, OpName, op
import ITensors.SiteTypes: @SiteType_str, @OpName_str
import ITensors.SiteTypes: siteinds
