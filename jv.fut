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

let augment_row [n] (costs: [n][n]f32) (row_dual: *[n]f32) (col_dual: [n]f32) (col_asgn: *[n]i64) (row: i64) : (*[n]f32, *[n]i64) =
  let cred = map2 (\cdu -> \row -> map2 (\ele -> \rdu -> ele - rdu - cdu) row row_dual) col_dual costs
  --let cred = map (\row -> map2 (\ele -> \rdu -> ele - rdu) row row_dual) costs
  let _ = trace cred
let row_asgn = other_asgn col_asgn
let _ = trace (map2 (\row -> \a -> let m = f32.minimum row in a == -1 || row[a] == m) cred row_asgn)
  let shortest = cred[row]
  let shortest_from = replicate n (-1)
  let found_best = replicate n false
  let (_, j) = trace (minidx shortest)
  let (_, shortest_from, j, found_best) = loop (shortest, shortest_from, j, found_best) while col_asgn[j] != -1 do
    let found_best = (copy found_best) with [j] = true
    let next_row = col_asgn[j]
    let mu = shortest[j]
    let shortest_from_j_asgn = map (+mu) cred[next_row]
    let is_shorter = map2 (<) shortest_from_j_asgn shortest

    let shortest = map2 f32.min shortest_from_j_asgn shortest
    let (s_i, s_u) = filter_by id is_shorter (zip (iota n) (replicate n j)) |> unzip
    let shortest_from = scatter shortest_from s_i s_u
    let (_, j) = minidx (map2 (\s -> \f -> if f then f32.inf else s) shortest found_best)
    in (shortest, shortest_from, j, found_best)
  
  let mu = cred[row, j]

  let from = shortest_from[j]
  let (col_asgn, row_dual, _, j) = loop (col_asgn, row_dual, from, j) while from != -1 do
    let new_asgn_row = col_asgn[from]
    let col_asgn = col_asgn with [j] = new_asgn_row
    let j = from
    let from = shortest_from[j]
    in (col_asgn, row_dual, from, j)

  let col_asgn = col_asgn with [j] = row
  
  --let offset = map2 (\d -> \asgn -> if asgn == -1 then 0 else (f32.max (d - mu) 0)) row_dual col_asgn
  --let offset = map2 (\d -> \asgn -> if asgn == -1 then 0 else d - mu) row_dual col_asgn
  --let offset = map2 (\d -> \f -> if f then d - mu else 0) row_dual found_best
let row_asgn = other_asgn col_asgn
let _ = trace (map2 (\row -> \a -> let m = f32.minimum row in a == -1 || row[a] == m) cred row_asgn)
  let offset = map2 (\d -> \f -> if f then (f32.max (d - mu) 0) else 0) row_dual found_best
  let row_dual = map2 (+) row_dual offset

  let _ = trace col_asgn
  in (row_dual, col_asgn)

-- returns new row_dual, col_asgn
--let augment_row [n] (costs: [n][n]f32) (row_dual: *[n]f32) (col_dual: [n]f32) (col_asgn: *[n]i64) (row: i64) : (*[n]f32, *[n]i64) =
--  let pred = replicate n row
--  let d = map2 (\c -> \du -> c - du - col_dual[row]) costs[row] row_dual
--  -- TODO is 0, SCAN is 1, READY is 2 
--  let sets = replicate n 0i64
--  let mu = -1.0
--  let exit_j = -1
--    let _ = trace ([]: []f32)
--  let (sets, pred, mu, d, exit_j) = loop (sets, pred, mu, d, exit_j) while exit_j == -1 do
--    let _ = trace d
--    let (mu, js) = filter_by (==0) sets d |> minidx
--    let _ = trace js
--    let exit_j = if col_asgn[js] == -1 then js else exit_j
--
--    let (sets, pred, mu, d, exit_j) = if exit_j == -1 then 
--      let i = col_asgn[js]
--      let sets = sets with [js] = 2
--      let offset = mu - col_dual[i]
--      let vals = map2 (\c -> \du -> offset + c - du) costs[i] row_dual
--      let smaller = map2 (\d -> \v -> v < d) d vals
--      let take = map2 (&&) smaller (map (==0) sets)
--        
--      let jt = filter_by id take (iota n)
--      let d_update = map (\j -> vals[j]) jt
--      let d = scatter d jt d_update
--      let (pred_i, pred_u) = map (\j -> (j, i)) jt |> unzip
--      let pred = scatter pred pred_i pred_u
--      
--      let cred_0 = filter_by id take (iota n) |> filter (\j -> vals[j] == mu)
--      let exit_js = cred_0 |> filter (\j -> col_asgn[j] == -1)
--      let exit_j = if null exit_js 
--        then exit_j
--        else head exit_js
--
--      let (sets_i, sets_u) = map (\j -> (j, 1)) cred_0 |> unzip
--      let sets = scatter sets sets_i sets_u
--
--      in (sets, pred, mu, d, exit_j)
--    else (sets, pred, mu, d, exit_j)
--
--    in (sets, pred, mu, d, exit_j)
--
--  let offset = map2 (\d -> \s -> if s == 2 then d - mu else 0) d sets
--  let row_dual = map2 (+) row_dual offset
--  
--  let row_asgn = other_asgn col_asgn
--  let j = exit_j
--  let i = -1
--  let (col_asgn, _, _, _) = loop (col_asgn, row_asgn, j, i) while i != row do 
--    let i = pred[j]
--    let col_asgn = col_asgn with [j] = i
--    let k = j
--    let j = row_asgn[i]
--    let row_asgn = row_asgn with [i] = k
--    in (col_asgn, row_asgn, j, i)
--  in (row_dual, col_asgn)

--entry main [n] (costs: *[n][n]f32) : *[n]i64 =
entry main [n] (costs: *[n][n]f32) : f32 =
  let (row_dual, row_asgn) = col_reduce costs
  let col_dual = row_reduce costs row_dual
  let col_asgn = other_asgn row_asgn
  let _ = trace col_asgn
  let (_, _, row_asgn, _) = loop (row_dual, col_dual, row_asgn, col_asgn) while 
    row_asgn |> any (\asgn -> asgn == -1) do
      let row = zip (iota n) row_asgn |> map (\(i, asgn) -> if asgn == -1 then i else -1) |> i64.maximum
      let (row_dual, col_asgn) = augment_row costs row_dual col_dual col_asgn row
      let col_dual = row_reduce costs row_dual

      in (row_dual, col_dual, other_asgn col_asgn, col_asgn)

  --in row_asgn
  in map2 (\i -> \row -> row[i]) row_asgn costs |> f32.sum
