let mn (m1:f32,i1:i64) (m2:f32,i2:i64) : (f32,i64) =
  if m1 < m2 then (m1,i1) else (m2,i2)

let minidx [n] (xs: [n]f32) : (f32,i64) =
  reduce mn (f32.inf,-1) (zip xs (indices xs))

let col_reduce [n] (costs: [n][n]f32) : (*[n]f32, *[n]i64) =
  let (mins, mins_idx): (*[n]f32, *[n]i64) = transpose costs |> map minidx |> unzip 
  in let is_min = indices mins_idx |> map (\i -> map (\m -> m==i) mins_idx)
  in let idx_array = replicate n (iota n)
  in let row_asgn = map2
    (map2 (\is_m -> \idx -> if is_m then idx else (-1)))
    is_min
    idx_array
    |> map i64.maximum
  in (copy mins, copy row_asgn)

-- does not assign
let row_reduce [n] (costs: [n][n]f32) (row_dual: [n]f32) : (*[n]f32) =
  map (\row -> map2 (-) row row_dual |> f32.minimum) costs

let other_asgn [n] (asgn: [n]i64) : *[n]i64 =
  scatter (replicate n (-1)) asgn (indices asgn)

let take_indices [n] 'a (idxs: [n]bool) (xs: [n]a) : []a =
  filter (\i -> idxs[i]) (indices idxs) |> map (\i -> xs[i])

let filter_by [n] 'a 'b (f: b -> bool) (filterer: [n]b) (to_filter: [n]a) : []a =
  zip to_filter filterer |> filter (\(_,b) -> f(b)) |> map (\(a,_) -> a)

let augment_row [n] 
(costs: [n][n]f32) 
(row_dual: *[n]f32) 
(col_dual: [n]f32) 
(col_asgn: *[n]i64) 
(row_asgn: *[n]i64) 
(row: i64) 
: (*[n]f32, *[n]i64, *[n]i64) = -- returns (row dual, col_asgn, row_asgn)
  let cred = map2 (\cdu -> \row -> map2 (\ele -> \rdu -> ele - rdu - cdu) row row_dual) col_dual costs
  let shortest = replicate n f32.inf
  let shortest_from = replicate n (-1)
  let todo = replicate n true
  let i = row
  let sink = -1
  let min = 0.0
  let (shortest, shortest_from, _, todo, min, sink) = 
    loop (shortest, shortest_from, i, todo, min, sink) while sink == -1 do
      let dist_from_i = map (+min) cred[i]
      let is_shorter = map2 (<) dist_from_i shortest
      let shortest = map2 f32.min dist_from_i shortest
      let (s_i, s_u) = filter_by id (map2 (&&) is_shorter todo) (zip (iota n) (replicate n i)) |> unzip
      let shortest_from = scatter shortest_from s_i s_u

      let (min, j) = minidx (map2 (\s -> \f -> if f then s else f32.inf) shortest todo)
      let todo = todo with [j] = false
      let asgn = col_asgn[j]
      let (sink, i) = if asgn == -1 then (j, i) else (sink, asgn)

      in (shortest, shortest_from, i, todo, min, sink)

  let row_dual = map3 (\du -> \s -> \f -> if f then du else du - min + s) row_dual shortest todo
  
  let j = sink
  let i = shortest_from[j]
  let col_asgn = col_asgn with [j] = i
  let temp = row_asgn[i]
  let row_asgn = row_asgn with [i] = j
  let j = temp
  let (col_asgn, row_asgn, _, _) = loop (col_asgn, row_asgn, j, i) while i != row do
    let i = shortest_from[j]
    let col_asgn = col_asgn with [j] = i
    let temp = row_asgn[i]
    let row_asgn = row_asgn with [i] = j
    let j = temp
    in (col_asgn, row_asgn, j, i)

  in (row_dual, col_asgn, row_asgn)
-- : (*[n]f32, *[n]i64, *[n]i64) = -- returns (row dual, col_asgn, row_asgn)

--entry main [n] (costs: *[n][n]f32) : *[n]i64 =
let jv [n] (costs: *[n][n]f32) : *[n]i64 = -- returns row assignments
  let (row_dual, row_asgn) = col_reduce costs
  let col_dual = row_reduce costs row_dual
  let col_asgn = other_asgn row_asgn
  let (_, _, row_asgn, _) = loop (row_dual, col_dual, row_asgn, col_asgn) while 
    row_asgn |> any (\asgn -> asgn == -1) do
      let row = zip (iota n) row_asgn |> map (\(i, asgn) -> if asgn == -1 then i else -1) |> i64.maximum
      let (row_dual, col_asgn, row_asgn) = augment_row costs row_dual col_dual col_asgn row_asgn row
      let col_dual = row_reduce costs row_dual

      in (row_dual, col_dual, row_asgn, col_asgn)

  in row_asgn
  --in map2 (\i -> \row -> row[i]) row_asgn costs |> f32.sum

--entry main [n] (costs: *[n][n]f32) (k: i64) : (f32, f32) = 
entry main [n] (costs: *[n][n]f32) : (f32, f32) = 
  --let initial_row_asgn = jv (copy costs)
  --let ms = replicate n costs 
  --  |> map3 (\j -> \i -> \c -> (copy c) with [i, j] = f32.inf) initial_row_asgn (iota n)
  --let row_asgns = map (\c -> jv (copy c)) ms
  --let first = map2 (\row -> \j -> row[j]) costs initial_row_asgn |> f32.sum
  --let second = map (\r_asgn -> map2 (\row -> \j -> row[j]) costs r_asgn |> f32.sum) row_asgns |> f32.minimum
  --in (first, second)
  --let costs = map (\row -> row ++ replicate n 0) costs -- turn to may not assign problem
  --let row_asgn = 

  
