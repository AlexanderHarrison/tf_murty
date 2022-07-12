import "radix_sort"

let mn (m1:f32,i1:i64) (m2:f32,i2:i64) : (f32,i64) =
  if m1 < m2 then (m1,i1) else (m2,i2)

let minidx [n] (xs: [n]f32) : (f32,i64) =
  reduce mn (f32.inf,-1) (zip xs (indices xs))

let row_reduce [n][m] (costs: [n][m]f32) : (*[n]f32, *[m]i64) =
  let (mins, mins_idx): (*[n]f32, *[n]i64) = costs |> map minidx |> unzip 
  let is_min = (iota m) |> map (\i -> map (==i) mins_idx) 
  let below_threshold = map (map (<=0.0)) costs |> transpose
  let possible_asgn = map2 (map2 (&&)) is_min below_threshold
  let idx_array = replicate m (iota n)
  let col_asgn = map2
    (map2 (\is_m -> \idx -> if is_m then idx else -1))
    possible_asgn
    idx_array
    |> map i64.maximum
  in (copy mins, copy col_asgn)

-- does not assign
let col_reduce [n][m] (costs: [n][m]f32) (col_dual: [n]f32) : (*[m]f32) =
  map (\col -> map2 (-) col col_dual |> f32.minimum) (transpose costs)
 
-- does not assign
let col_dual_by_reduce [n][m] (costs: [n][m]f32) (row_dual: [m]f32) : (*[n]f32) =
  map (\row -> map2 (-) row row_dual |> f32.minimum) costs

let other_asgn [n][m] (asgn: [n]i64) : *[m]i64 =
  scatter (replicate m (-1)) asgn (indices asgn)

let take_indices [n] 'a (idxs: [n]bool) (xs: [n]a) : []a =
  filter (\i -> idxs[i]) (indices idxs) |> map (\i -> xs[i])

let filter_by [n] 'a 'b (f: b -> bool) (filterer: [n]b) (to_filter: [n]a) : []a =
  zip to_filter filterer |> filter (\(_,b) -> f(b)) |> map (\(a,_) -> a)

let augment_row [n][m]
(costs: [n][m]f32) 
(row_dual: *[m]f32) 
(col_asgn: *[m]i64) 
(row_asgn: *[n]i64) 
(row: i64) 
: (*[m]f32, *[m]i64, *[n]i64) = -- returns (row dual, col_asgn, row_asgn)
  let err = map (\row -> map2 (-) row row_dual) costs
    |> map2 (\r_a -> \row -> r_a == -1 || let (_, c) = minidx row in c == r_a) row_asgn 
    |> any not

  let _ = if err then 
    trace costs else costs

  let dist_to_col = map2 (-) costs[row] row_dual
  let pathback = replicate m row
  
  let (shortest_path, col) = minidx dist_to_col
  let unused_cols = replicate m true

  let minmissval = f32.inf
  let minmissi = row
  let i = row
  let counter = 0

  let (_, pathback, _, shortest_path, unused_cols, col, _, minmissi, row_dual, i) = 
    loop (counter, pathback, dist_to_col, shortest_path, unused_cols, col, minmissval, minmissi, row_dual, i)
    while col != -1 && col_asgn[col] != -1 && counter <= n do
      let i = col_asgn[col]
      let prev = row_dual[col]
      let row_dual = row_dual with [col] = prev + shortest_path
      let unused_cols = unused_cols with [col] = false
      let u1 = costs[i, col] - prev - shortest_path

      let (minmissval, minmissi) = if -u1 < minmissval then (-u1, i) else (minmissval, minmissi)

      let row_cred = map2 (-) costs[i] row_dual |> map (\n -> n-u1)
      let shorter = map2 (<) row_cred dist_to_col
      let take = map2 (&&) shorter unused_cols
      let js = filter_by id take (iota m)
      let pathback = scatter pathback js (map (\_ -> i) js)
      let (d_js, new_dists) = filter_by id take (zip (iota m) row_cred) |> unzip
      let dist_to_col = scatter dist_to_col d_js new_dists

      let (shortest_path, col) = minidx (
        (map2 (\un -> \d -> if un then d else f32.inf) unused_cols dist_to_col))
  
      in (counter+1,pathback, dist_to_col, shortest_path, unused_cols, col, minmissval, minmissi, row_dual, i)

  let i = if col != -1 && col_asgn[col] == -1 then col_asgn[col] else i
  let prev = row_asgn[minmissi]
  let (i, col, row_asgn) = if col == -1 
    then (minmissi, prev, row_asgn with [minmissi] = -1)
    else (i, col, row_asgn)

  let (_, col_asgn, row_asgn, _) = loop (col, col_asgn, row_asgn, i) while i != row do
    let i = pathback[col]
    let col_asgn = col_asgn with [col] = i
    let k = col
    let col = row_asgn[i]
    let row_asgn = row_asgn with [i] = k
    in (col, col_asgn, row_asgn, i)

  let offset = map (\un -> if un then 0.0 else shortest_path) unused_cols
  let row_dual = map2 (-) row_dual offset

  in (row_dual, col_asgn, row_asgn)

let jv [n][m] (costs: [n][m]f32) : ([n]i64, [m]f32) =
  let (_col_dual, col_asgn) = row_reduce costs 
  let row_asgn = other_asgn col_asgn
  let row_dual = replicate m 0.0 -- maybe?
  let unassigned = filter_by (== -1) (copy row_asgn) (iota n)
  let (row_dual, row_asgn, _) = loop (row_dual, row_asgn, col_asgn) for row in unassigned do
      let (row_dual, col_asgn, row_asgn) = augment_row costs row_dual col_asgn row_asgn row
      in (row_dual, row_asgn, col_asgn)
  in (row_asgn, row_dual)

let score [n][m] (costs: [n][m]f32) (row_asgn: [n]i64) : f32 =
  map2 (\j -> \row -> if j == -1 then 0.0 else row[j]) row_asgn costs |> f32.sum

let gen_murty_costs [n][m] (costs: [n][m]f32) (row_asgn: [n]i64) : *[n][n][m]f32 =
  map (\i -> map3
    (
      \i2 -> \asgn -> \row -> if i2 < i 
        then map2 (\j -> \ele -> if j==asgn then ele else f32.inf) (iota m) row
        else if i == i2 then copy row with [asgn] = f32.inf
        else row
    ) (iota n) row_asgn costs
    ) (iota n)

let murty [n][m] (costs: [n][m]f32) (k: i64) : [k][n]i64 =
  let (least_row_asgn, row_dual) = jv costs

  let filler = replicate (k-1) (
    replicate m 0.0, -- row duals
    replicate n (replicate m f32.inf), -- costs
    iota n -- asgns
  )

  let (least_row_duals, least_costs, least_row_asgns) =
    concat_to k [(row_dual, costs, least_row_asgn)] filler
    |> unzip3

  let (_, _, least_row_asgns) = loop (least_row_duals, least_costs, least_row_asgns) for i < (k-1) do
    let least_row_dual = least_row_duals[i]
    let least_cost = least_costs[i]
    let least_row_asgn = least_row_asgns[i]
    let new_costs = gen_murty_costs least_cost least_row_asgn
      |> filter (all (\r -> !(all (==f32.inf) r)))

    let (new_row_asgns, new_row_duals) = map jv new_costs |> unzip
    let new_data = zip3 new_row_duals new_costs new_row_asgns
    let to_sort = (zip3 least_row_duals least_costs least_row_asgns) ++ new_data
    let (least_row_duals, least_costs, least_row_asgns) = 
      radix_sort_float_by_key (\(_, c, r) -> score c r) f32.num_bits f32.get_bit to_sort
      |> take k |> unzip3

    in (least_row_duals, least_costs, least_row_asgns)
  in least_row_asgns

entry main [n][m] (costs: [n][m]f32) (k: i64) : [k][n]i64 = 
  let r = n+m
  let augmented_costs: [n][r]f32 = map2 (\i -> \row -> concat_to r row (replicate n f32.inf with [i] = 0)) (iota n) costs
  let augemented_row_asgns = murty augmented_costs k
  in map (map (\r_a -> if r_a >= m then -1 else r_a)) augemented_row_asgns
