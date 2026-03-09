# restrictedsparse/test_restrictedsparse.jl
using Random
using LinearAlgebra
include("restrictedsparse.jl")
using .RestrictedSparse: restrictedsparse, from_dense, to_dense

function make_dense_example(dims::NTuple{N,Int}; diag_pairs=Tuple{Int,Int}[]) where {N}
  # Build a dense array that respects:
  # - optional diagonal constraints on prefix axes
  # - restricted-perm on last 2 axes: for each (prefix,row) choose <=1 col
  T = Float64
  A = zeros(T, dims)

  P = N - 2
  rows = dims[N-1]
  cols = dims[N]

  # Iterate over all prefix indices (small dims!)
  prefix_dims = P > 0 ? ntuple(i -> dims[i], Val(P)) : ()
  if P > 0
    for Iprefix in CartesianIndices(prefix_dims)
      # enforce diagonal constraints by only filling those that satisfy
      ok = true
      for (a,b) in diag_pairs
        if Iprefix[a] != Iprefix[b]
          ok = false
          break
        end
      end
      ok || continue

      for row in 1:rows
        # choose either none or one col
        if rand() < 0.3
          continue
        end
        col = rand(1:cols)
        v = randn()

        I = ntuple(i -> begin
          if i <= P
            Iprefix[i]
          elseif i == N-1
            row
          else
            col
          end
        end, Val(N))
        A[I...] = v
      end
    end
  else
    # N==2: just restricted perm matrix
    for row in 1:rows
      rand() < 0.3 && continue
      col = rand(1:cols)
      A[row,col] = randn()
    end
  end

  return A
end

function main()
  Random.seed!(0)

  # Example: N=4 with constraint (1==2), restricted perm on (3,4) treated as (row=3,col=4)
  dims = (3,3,4,5)
  diag_pairs = [(1,2)]
  A = make_dense_example(dims; diag_pairs=diag_pairs)

  rs = from_dense(A; diag_pairs=diag_pairs, atol=0.0)
  B = to_dense(rs)

  println("max|A-B| = ", maximum(abs.(A .- B)))
  @assert maximum(abs.(A .- B)) == 0.0

  # Example: N=6 with constraints (1==2) and (3==4), last two are (5,6)
  dims6 = (2,2,3,3,4,4)
  diag_pairs6 = [(1,2),(3,4)]
  A6 = make_dense_example(dims6; diag_pairs=diag_pairs6)

  rs6 = from_dense(A6; diag_pairs=diag_pairs6)
  B6 = to_dense(rs6)
  println("max|A6-B6| = ", maximum(abs.(A6 .- B6)))
  @assert maximum(abs.(A6 .- B6)) == 0.0

  println("OK")
end

main()