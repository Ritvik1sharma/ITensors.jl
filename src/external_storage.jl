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

# module ITensors

# """
# ExternalStorage holds an arbitrary payload `data`.
# Index bookkeeping is delegated to `inds(data)`.
# """
# struct ExternalStorage{S}
#   data::S
# end

# # === Required so `inds(T::ITensor) = inds(T.tensor)` works ===
# @inline inds(es::ExternalStorage) = inds(es.data)
# @inline ninds(es::ExternalStorage) = length(inds(es))
# @inline order(es::ExternalStorage) = ninds(es)

# # Optional (sometimes useful)
# @inline has_external_storage(T::ITensor) = (T.tensor isa ExternalStorage)
# @inline get_external_storage(T::ITensor) = (T.tensor::ExternalStorage).data

# # Constructor hook: create an ITensor whose payload is ExternalStorage(data)
# # IMPORTANT: this assumes your ITensor stores `tensor` as the payload and
# # that your existing ITensor logic gets inds from `inds(T.tensor)`.
# @inline function _itensor_from_external_storage(data)
#   return _ITensor(ExternalStorage{typeof(data)}(data))
# end

# # Hook that external packages (SparseBackends) implement.
# # Must return a new payload (e.g. WrappedTensorTypes) whose `inds(payload)` are correct.
# function _contract_external_storage(dataA, dataB; kwargs...)
#   error("No external-storage contraction defined for $(typeof(dataA)) × $(typeof(dataB))")
# end

# # -----------------------------
# # Multiply / contract dispatch
# # -----------------------------

# import Base: *

# # Wrap existing multiply:
# function *(A::ITensor, B::ITensor)
#   if has_external_storage(A) || has_external_storage(B)
#     # If one side is external, we want to route through the hook.
#     # Make both sides "payload objects":
#     dataA = has_external_storage(A) ? get_external_storage(A) : A
#     dataB = has_external_storage(B) ? get_external_storage(B) : B

#     # Only handle the external-external case here (your requested behavior).
#     # If you want external × dense too, implement it in the hook or wrap dense first.
#     if !(dataA isa ITensor) && !(dataB isa ITensor)
#       dataC = _contract_external_storage(dataA, dataB)
#       return _itensor_from_external_storage(dataC)
#     end
#     # else: fall through to normal ITensors multiplication
#   end

#   # Fallback: call the "next most specific" existing method
#   return invoke(*, Tuple{ITensor,ITensor}, A, B)
# end

# # If ITensors has a `contract(A::ITensor,B::ITensor)` already, wrap it similarly.
# # If not, define it and fall back to `A * B` or whatever your fork uses.
# function contract(A::ITensor, B::ITensor; kwargs...)
#   if has_external_storage(A) || has_external_storage(B)
#     dataA = has_external_storage(A) ? get_external_storage(A) : A
#     dataB = has_external_storage(B) ? get_external_storage(B) : B
#     if !(dataA isa ITensor) && !(dataB isa ITensor)
#       dataC = _contract_external_storage(dataA, dataB; kwargs...)
#       return _itensor_from_external_storage(dataC)
#     end
#   end
#   return invoke(contract, Tuple{ITensor,ITensor}, A, B; kwargs...)
# end

# end # module


# # module ITensors

# # """
# # A minimal payload that can live inside ITensor.tensor but still
# # participate in index bookkeeping.
# # """
# # struct ExternalStorage{S,N}
# #   data::S
# #   inds::NTuple{N,Index}
# # end


# # # Make existing `inds(T::ITensor) = inds(T.tensor)` work
# # @inline inds(es::ExternalStorage) = es.inds
# # @inline ninds(es::ExternalStorage{S,N}) where {S,N} = N
# # @inline order(es::ExternalStorage{S,N}) where {S,N} = N

# # @inline function _itensor_from_external_storage(data, inds::Vararg{Index,N}) where {N}
# #   return _ITensor(ExternalStorage{typeof(data),N}(data, Tuple(inds)))
# # end

# # @inline has_external_storage(T::ITensor) = (T.tensor isa ExternalStorage)
# # @inline get_external_storage(T::ITensor) = (T.tensor::ExternalStorage).data
# # @inline get_external_inds(T::ITensor)    = (T.tensor::ExternalStorage).inds


# # function _contract_external_storage(dataA, indsA, dataB, indsB)
# #   error("No external-storage contraction defined for $(typeof(dataA)) × $(typeof(dataB))")
# # end

# # import Base: *

# # function *(A::ITensor, B::ITensor)
# #   if has_external_storage(A) && has_external_storage(B)
# #     dataA = get_external_storage(A)
# #     dataB = get_external_storage(B)
# #     indsA = Tuple(inds(A))
# #     indsB = Tuple(inds(B))
# #     dataC, indsC = _contract_external_storage(dataA, indsA, dataB, indsB)
# #     return _itensor_from_external_storage(dataC, indsC...)
# #   end
# #   error("No fallback ITensor multiplication defined. Patch existing `*(::ITensor,::ITensor)` instead of using _mul_fallback.")
# # end

# # function contract(A::ITensor, B::ITensor)
# #   if has_external_storage(A) && has_external_storage(B)
# #     dataA = get_external_storage(A)
# #     dataB = get_external_storage(B)
# #     indsA = Tuple(inds(A))
# #     indsB = Tuple(inds(B))
# #     dataC, indsC = _contract_external_storage(dataA, indsA, dataB, indsB)
# #     return _itensor_from_external_storage(dataC, indsC...)
# #   end
# #   error("No fallback ITensor contract defined. Patch existing `contract(::ITensor,::ITensor)` instead of using _contract_fallback.")
# # end



# # end # module
