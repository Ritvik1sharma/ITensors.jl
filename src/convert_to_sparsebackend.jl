using SparseBackends

# bra_plev / ket_plev = desired prime levels of the two "Site" indices
function mpo_axes_itensor(T::ITensor, bra_plev::Int, ket_plev::Int)
  is = collect(inds(T))
  s_bra = only([I for I in is if ITensors.hastags(I, "Site") && ITensors.hasplev(I, bra_plev)])
  s_ket = only([I for I in is if ITensors.hastags(I, "Site") && ITensors.hasplev(I, ket_plev)])
  links = [I for I in is if ITensors.hastags(I, "Link")]
  return s_bra, s_ket, links
end

function convert_to_array(T::ITensor)
    bra, ket, links = mpo_axes_itensor(T, 1, 0)
    array = Array(T, bra, ket, links...)
    labels = Symbol[Symbol("s_p$bra_plev"), Symbol("s_p$ket_plev")]
    append!(labels, Symbol.("pL", 1:length(links)))
    map = Dict(l => i for (i,l) in pairs(labels))
    return array, labels, map
end

function convert_to_coo(T::ITensor)
    bra, ket, links = mpo_axes_itensor(T, 1, 0)
    array = Array(T, bra, ket, links...)
    coo = SparseBackends.coo_from_dense(array; atol=1e-12, rtol=0.0)
    labels = Symbol[Symbol("s_p$bra_plev"), Symbol("s_p$ket_plev")]
    append!(labels, Symbol.("pL", 1:length(links)))
    map = Dict(l => i for (i,l) in pairs(labels))
    return coo, labels, map
end

function create_blocksparse_output(::Type{T}, output_dims::NTuple{N,Int}, denseLinks::Int) where {T,N}
    A = zeros(T, output_dims...)          # Array{T,N}
    return SparseBackends.blocksparse_from_dense(A, Val(denseLinks))
end