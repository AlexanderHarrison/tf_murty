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
    (map2 (\is_m -> \idx -> if is_m then idx else (-1)))
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
(col_dual: [n]f32) 
(col_asgn: *[m]i64) 
(row_asgn: *[n]i64) 
(row: i64) 
: (*[m]f32, *[m]i64, *[n]i64) = -- returns (row dual, col_asgn, row_asgn)
  let shortest_path = 0
  :!

entry main [n][m] (costs: *[n][m]f32) : f32 =
--entry main [n][m] (costs: *[n][m]f32) (asgn_row_cost: *[m]f32) (asgn_col_cost: *[n]f32) : f32 =
--  let costs = map2 (\row -> \a -> map2 (\r -> \b -> r - a - b) row asgn_row_cost) costs asgn_col_cost
  let (col_dual, col_asgn) = row_reduce costs 
  let (row_dual) = col_reduce costs col_dual
  let row_asgn = other_asgn col_asgn
  let unassigned = filter_by (== -1) (copy row_asgn) (iota n)
          let _ = trace row_asgn
          --let cred = map2 (\cdu -> \row -> map2 (\ele -> \rdu -> ele - rdu - cdu) row row_dual) col_dual costs
          --let mins = map f32.minimum cred
          --let _ = trace ( map3 (\row -> \j -> \m -> j == -1 || row[j] == m) cred row_asgn mins |> all id)
  let (_, _, row_asgn, _) = loop (row_dual, col_dual, row_asgn, col_asgn) for row in unassigned do
      --let row = zip (iota n) row_asgn |> map (\(i, asgn) -> if asgn == -1 then i else -1) |> i64.maximum
      let (row_dual, col_asgn, row_asgn) = augment_row costs row_dual col_dual col_asgn row_asgn row
      let col_dual = col_dual_by_reduce costs row_dual
          let _ = trace row_asgn
          --let cred = map2 (\cdu -> \row -> map2 (\ele -> \rdu -> ele - rdu - cdu) row row_dual) col_dual costs
          --let mins = map f32.minimum cred
          --let _ = trace ( map3 (\row -> \j -> \m -> j == -1 || row[j] == m) cred row_asgn mins |> all id)
      in (row_dual, col_dual, row_asgn, col_asgn)

  in map2 (\i -> \row -> if i == -1 then 0.0 else row[i]) row_asgn costs |> f32.sum

  --let cred = map2 (\cdu -> \row -> map2 (\ele -> \rdu -> ele - rdu - cdu) row row_dual) col_dual costs
--  let cred = costs
--  let shortest = replicate m f32.inf
--  let shortest_from = replicate m (-1)
--  let todo = replicate m true
--  let i = row
--  let sink = -1
--  let min = 0.0
--  let (shortest, shortest_from, _, todo, min, sink) = 
--    loop (shortest, shortest_from, i, todo, min, sink) while sink == -1 do
--      let dist_from_i = map (+min) cred[i]
--      let is_shorter = map2 (<) dist_from_i shortest
--      let shortest = map2 f32.min dist_from_i shortest
--      let (s_i, s_u) = filter_by id (map2 (&&) is_shorter todo) (zip (iota m) (replicate m i)) |> unzip
--      let shortest_from = scatter shortest_from s_i s_u
--
--      let (min, j) = minidx (map2 (\s -> \f -> if f then s else f32.inf) shortest todo)
--      let todo = todo with [j] = false
--      let asgn = col_asgn[j]
--      let min = 
--      let (sink, i) = if asgn == -1 then (j, i) else (sink, asgn)
--
--      in (shortest, shortest_from, i, todo, min, sink)
--
--  in if min < 0.0 then
--    let row_dual = map3 (\du -> \s -> \f -> if f then du else du - min + s) row_dual shortest todo
--    
--    let j = sink
--    let i = shortest_from[j]
--    let col_asgn = col_asgn with [j] = i
--    let temp = row_asgn[i]
--    let row_asgn = row_asgn with [i] = j
--    let j = temp
--    let (col_asgn, row_asgn, _, _) = loop (col_asgn, row_asgn, j, i) while i != row do
--      let i = shortest_from[j]
--      let col_asgn = col_asgn with [j] = i
--      let temp = row_asgn[i]
--      let row_asgn = row_asgn with [i] = j
--      let j = temp
--      in (col_asgn, row_asgn, j, i)
--
--    in (row_dual, col_asgn, row_asgn)
--  else let _ = trace min in (row_dual, col_asgn, row_asgn)
