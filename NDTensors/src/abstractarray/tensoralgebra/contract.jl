using LinearAlgebra: BlasFloat
using .Expose: expose

# TODO: Delete these exports
export backend_auto, backend_blas, backend_generic

@eval struct GemmBackend{T}
    (f::Type{<:GemmBackend})() = $(Expr(:new, :f))
end
GemmBackend(s) = GemmBackend{Symbol(s)}()
macro GemmBackend_str(s)
    return :(GemmBackend{$(Expr(:quote, Symbol(s)))})
end

const gemm_backend = Ref(:Auto)
function backend_auto()
    return gemm_backend[] = :Auto
end
function backend_blas()
    return gemm_backend[] = :BLAS
end
function backend_generic()
    return gemm_backend[] = :Generic
end

@inline function auto_select_backend(
        ::Type{<:StridedVecOrMat{<:BlasFloat}},
        ::Type{<:StridedVecOrMat{<:BlasFloat}},
        ::Type{<:StridedVecOrMat{<:BlasFloat}},
    )
    return GemmBackend(:BLAS)
end

@inline function auto_select_backend(
        ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}, ::Type{<:AbstractVecOrMat}
    )
    return GemmBackend(:Generic)
end

function _gemm!(
        tA, tB, alpha, A::TA, B::TB, beta, C::TC
    ) where {TA <: AbstractVecOrMat, TB <: AbstractVecOrMat, TC <: AbstractVecOrMat}
    return if gemm_backend[] == :Auto
        _gemm!(auto_select_backend(TA, TB, TC), tA, tB, alpha, A, B, beta, C)
    else
        _gemm!(GemmBackend(gemm_backend[]), tA, tB, alpha, A, B, beta, C)
    end
end

# BLAS matmul
function _gemm!(
        ::GemmBackend{:BLAS},
        tA,
        tB,
        alpha,
        A::AbstractVecOrMat,
        B::AbstractVecOrMat,
        beta,
        C::AbstractVecOrMat,
    )
    #@timeit_debug timer "BLAS.gemm!" begin
    return BLAS.gemm!(tA, tB, alpha, A, B, beta, C)
    #end # @timeit
end

# generic matmul
function _gemm!(
        ::GemmBackend{:Generic},
        tA,
        tB,
        alpha::AT,
        A::AbstractVecOrMat,
        B::AbstractVecOrMat,
        beta::BT,
        C::AbstractVecOrMat,
    ) where {AT, BT}
    mul!(
        expose(C),
        expose(tA == 'T' ? transpose(A) : A),
        expose(tB == 'T' ? transpose(B) : B),
        alpha,
        beta,
    )
    return C
end

# Non-trivial permutation
function _contract_scalar_perm!(
        Rᵃ::AbstractArray{ElR}, Tᵃ::AbstractArray, perm, α, β = zero(ElR)
    ) where {ElR}
    if iszero(β)
        if iszero(α)
            fill!(Rᵃ, 0)
        else
            Rᵃ = permutedims!!(Rᵃ, Tᵃ, perm, (r, t) -> α * t)
        end
    elseif isone(β)
        if iszero(α)
            # Rᵃ .= Rᵃ
            # No-op
        else
            Rᵃ = permutedims!!(Rᵃ, Tᵃ, perm, (r, t) -> r + α * t)
        end
    else
        if iszero(α)
            # Rᵃ .= β .* Rᵃ
            LinearAlgebra.scal!(length(Rᵃ), β, Rᵃ, 1)
        else
            Rᵃ .= α .* permutedims(expose(Tᵃ), perm) .+ β .* Rᵃ
        end
    end
    return Rᵃ
end

const _NDT_PROFILE_IO = Ref{Union{Nothing,IO}}(nothing)
const _NDT_PROFILE_INIT = Ref{Bool}(false)
function _ndt_profile_io()
    if !_NDT_PROFILE_INIT[]
        _NDT_PROFILE_INIT[] = true
        path = get(ENV, "SB_PERMUTE_PROFILE", "")
        if !isempty(path)
            _NDT_PROFILE_IO[] = open(path, "a")
            atexit() do
                io = _NDT_PROFILE_IO[]
                if io !== nothing; close(io); _NDT_PROFILE_IO[] = nothing; end
            end
        end
    end
    return _NDT_PROFILE_IO[]
end

function _contract!(
        CT::AbstractArray{El, NC},
        AT::AbstractArray{El, NA},
        BT::AbstractArray{El, NB},
        props::ContractionProperties,
        α::Number = one(El),
        β::Number = zero(El),
    ) where {El, NC, NA, NB}
    _ndt_io = _ndt_profile_io()
    _ndt_active = _ndt_io !== nothing
    _ndt_t_permA = 0.0; _ndt_t_permB = 0.0; _ndt_t_permC_in = 0.0; _ndt_t_permC_out = 0.0
    _ndt_t_gemm = 0.0
    tA = 'N'
    if props.permuteA
        _t0 = _ndt_active ? time_ns() : UInt64(0)
        Ap = permutedims(expose(AT), props.PA)
        _ndt_active && (_ndt_t_permA = (time_ns() - _t0) / 1e9)
        AM = transpose(reshape(Ap, (props.dmid, props.dleft)))
    else
        if Atrans(props)
            AM = transpose(reshape(AT, (props.dmid, props.dleft)))
        else
            AM = reshape(AT, (props.dleft, props.dmid))
        end
    end

    tB = 'N'
    if props.permuteB
        _t0 = _ndt_active ? time_ns() : UInt64(0)
        Bp = permutedims(expose(BT), props.PB)
        _ndt_active && (_ndt_t_permB = (time_ns() - _t0) / 1e9)
        BM = reshape(Bp, (props.dmid, props.dright))
    else
        if Btrans(props)
            BM = transpose(reshape(BT, (props.dright, props.dmid)))
        else
            BM = reshape(BT, (props.dmid, props.dright))
        end
    end

    if props.permuteC
        if β ≠ 0
            _t0 = _ndt_active ? time_ns() : UInt64(0)
            CM = reshape(permutedims(expose(CT), invperm(props.PC)), (props.dleft, props.dright))
            _ndt_active && (_ndt_t_permC_in = (time_ns() - _t0) / 1e9)
        else
            CM = reshape(copy(CT), (props.dleft, props.dright))
        end
    else
        if Ctrans(props)
            CM = transpose(reshape(CT, (props.dright, props.dleft)))
        else
            CM = reshape(CT, (props.dleft, props.dright))
        end
    end

    _t0 = _ndt_active ? time_ns() : UInt64(0)
    CM = mul!!(CM, AM, BM, El(α), El(β))
    _ndt_active && (_ndt_t_gemm = (time_ns() - _t0) / 1e9)

    if props.permuteC
        Cr = reshape(CM, props.newCrange)
        _t0 = _ndt_active ? time_ns() : UInt64(0)
        CT .= permutedims(expose(Cr), props.PC)
        _ndt_active && (_ndt_t_permC_out = (time_ns() - _t0) / 1e9)
    end

    if _ndt_active
        _ndt_site = get(ENV, "SB_IN_POSITION", "0") == "1" ? "position" : "matvec"
        _ndt_total = _ndt_t_permA + _ndt_t_permB + _ndt_t_permC_in + _ndt_t_permC_out + _ndt_t_gemm
        # permute_A_s holds permuteA, permute_B_s holds permuteB+permuteC_in+permuteC_out
        # to fit our TSV schema. Analysis script will recognize kernel=denseH_internal.
        _ndt_t_pB_total = _ndt_t_permB + _ndt_t_permC_in + _ndt_t_permC_out
        println(_ndt_io, _ndt_site, "\tdenseH_internal\t",
                NA, "\t", NB, "\t", NC, "\t-\t",
                _ndt_t_permA, "\t", _ndt_t_pB_total, "\t", _ndt_t_gemm, "\t", _ndt_total, "\t-\t-\t-\t",
                "permA=", props.permuteA ? 1 : 0,
                ";permB=", props.permuteB ? 1 : 0,
                ";permC=", props.permuteC ? 1 : 0,
                ";dmid=", props.dmid,
                ";dleft=", props.dleft,
                ";dright=", props.dright,
                "\t-")
    end

    return CT
end
