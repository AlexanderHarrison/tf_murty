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

let other_asgn [n] (asgn: [n]i64) : *[n]i64 =
  scatter (replicate n (-1)) asgn (indices asgn)

let take_indices [n] 'a (idxs: [n]bool) (xs: [n]a) : []a =
  filter (\i -> idxs[i]) (indices idxs) |> map (\i -> xs[i])

let filter_by [n] 'a 'b (f: b -> bool) (filterer: [n]b) (to_filter: [n]a) : []a =
  zip to_filter filterer |> filter (\(_,b) -> f(b)) |> map (\(a,_) -> a)

-- returns new row_dual, col_asgn
let augment_row [n] (costs: [n][n]f32) (row_dual: *[n]f32) (col_dual: [n]f32) (col_asgn: *[n]i64) (row: i64) : (*[n]f32, *[n]i64) =
  let pred = replicate n row
  let d = map2 (\c -> \du -> c - du - col_dual[row]) costs[row] row_dual
  -- TODO is 0, SCAN is 1, READY is 2 
  let sets = replicate n 0i64
  let mu = -1.0
  let exit_j = -1
    let _ = trace ([]: []f32)
  let (sets, pred, mu, d, exit_j) = loop (sets, pred, mu, d, exit_j) while exit_j == -1 do
    let _ = trace sets
    let _ = trace pred
    let _ = trace mu
    let _ = trace d

    let (mu, js) = filter_by (==0) sets d |> minidx
    let exit_j = if col_asgn[js] == -1 then js else exit_j

    let (sets, pred, mu, d, exit_j) = if exit_j == -1 then 
      let i = col_asgn[js]
      let sets = sets with [js] = 2
      let offset = mu - col_dual[i]
      let vals = map2 (\c -> \du -> offset + c - du) costs[i] row_dual
      let smaller = map2 (\d -> \v -> v < d) d vals
      let take = map2 (&&) smaller (map (==0) sets)
        
      let jt = filter_by id take (iota n)
      let d_update = map (\j -> vals[j]) jt
      let d = scatter d jt d_update
      let (pred_i, pred_u) = map (\j -> (j, i)) jt |> unzip
      let pred = scatter pred pred_i pred_u
      
      let cred_0 = filter_by id take (iota n) |> filter (\j -> vals[j] == mu)
      let exit_js = cred_0 |> filter (\j -> col_asgn[j] == -1)
      let exit_j = if null exit_js 
        then exit_j
        else head exit_js

      let (sets_i, sets_u) = map (\j -> (j, 1)) cred_0 |> unzip
      let sets = scatter sets sets_i sets_u

      in (sets, pred, mu, d, exit_j)
    else (sets, pred, mu, d, exit_j)

    in (sets, pred, mu, d, exit_j)

  let offset = map2 (\d -> \s -> if s == 2 then d - mu else 0) d sets
  let row_dual = map2 (+) row_dual offset
  
  let row_asgn = other_asgn col_asgn
  let j = exit_j
  let i = -1
  let (col_asgn, _, _, _) = loop (col_asgn, row_asgn, j, i) while i != row do 
    let i = pred[j]
    let col_asgn = col_asgn with [j] = i
    let k = j
    let j = row_asgn[i]
    let row_asgn = row_asgn with [i] = k
    in (col_asgn, row_asgn, j, i)
  in (row_dual, col_asgn)

--entry main [n] (costs: *[n][n]f32) : *[n]i64 =
entry main [n] (costs: *[n][n]f32) : f32 =
  let (row_dual, row_asgn) = col_reduce costs
  in let col_dual = replicate n 0
  in let col_asgn = other_asgn row_asgn
  in let (_, row_asgn, _) = loop (row_dual, row_asgn, col_asgn) while 
    row_asgn |> any (\asgn -> asgn == -1) do
      let row = zip (iota n) row_asgn |> map (\(i, asgn) -> if asgn == -1 then i else -1) |> i64.maximum
      in let (row_dual, col_asgn) = augment_row costs row_dual col_dual col_asgn row
      in (row_dual, other_asgn col_asgn, col_asgn)

  in let _ = trace row_asgn
  --in row_asgn
  in map2 (\i -> \row -> row[i]) row_asgn costs |> f32.sum
