# external_storage.jl
# NOTE: This file is included inside `module ITensors` already.
# Do NOT write `module ITensors` here.

"""
ExternalStorage holds an arbitrary payload `data`.
Index bookkeeping is delegated to `inds(data)`.
"""
struct ExternalStorage{S}
  data::S
end

# Make existing `inds(T::ITensor) = inds(T.tensor)` work by defining inds on ExternalStorage
@inline inds(es::ExternalStorage) = inds(es.data)
@inline ninds(es::ExternalStorage) = length(inds(es))
@inline order(es::ExternalStorage) = ninds(es)

@inline has_external_storage(T::ITensor) = (T.tensor isa ExternalStorage)
@inline get_external_storage(T::ITensor) = (T.tensor::ExternalStorage).data

# Delegate copy to the payload (needed by deepcopy(::MPS) used by excited-state DMRG).
Base.copy(es::ExternalStorage) = ExternalStorage(copy(es.data))

Base.deepcopy(es::ExternalStorage) = ExternalStorage(deepcopy(es.data))

# Constructor hook: build an ITensor whose payload is ExternalStorage(data)
@inline _itensor_from_external_storage(data) = _ITensor(ExternalStorage{typeof(data)}(data))

"""
Hook: external packages (SparseBackends) extend this.
Should return a payload `dataC` whose inds are accessible as `inds(dataC)`.
"""
function _contract_external_storage(dataA, dataB; kwargs...)
  error("No external-storage contraction defined for $(typeof(dataA)) × $(typeof(dataB))")
end

function external_dag(T::ITensor)
  error("No external-storage dag defined for $(typeof(T.tensor))")
end

function _dims(es::ExternalStorage)
    try
        return _dims(es.data)
    catch
        return "unknown dims for $(typeof(es.data))"
    end
end


import Base: show, summary

function summary(io::IO, es::ExternalStorage)
    print(io, "ExternalStorage(", typeof(es.data), ")", " with ", inds(es), " and dims ", _dims(es.data), " and payload ", es.data)
end

function show(io::IO, es::ExternalStorage)
    summary(io, es)
end

function show(io::IO, ::MIME"text/plain", es::ExternalStorage)
    summary(io, es)
    # show inds if the payload supports inds()
    try
        println(io, "  payload dims: ", (es.data.dims))
        println(io, "  inds: ", inds(es))
        println(io, " payload ", es.data)
    catch
        # ok: some payloads might not implement inds
    end
end

function dim(es::ExternalStorage)
    d = _dims(es)
    d isa Integer && return Int(d)
    d isa Tuple && return prod(Int, d)
    return 1  # fallback: unknown payload
end

