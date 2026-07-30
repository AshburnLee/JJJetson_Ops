# `moe_dispatch.cu` 逻辑流程 用小case 走一遍

## 常量 case

`num_tokens` = 5
`hidden_size` = 4
`num_experts` = 3
`top_k` = 2
`num_routes` = `num_tokens` × `top_k` = 10
scatter（本走读假定）：按 `r = 0…9` 顺序处理，与串行等价 `atomicAdd` 的结果一致。


## 0 `r ↔ (t,k)`

`t = r // top_k`, `k = r % top_k`

~~~
0  `t`=0 | `k`=0
1  `t`=0 | `k`=1
2  `t`=1 | `k`=0
3  `t`=1 | `k`=1
4  `t`=2 | `k`=0
5  `t`=2 | `k`=1
6  `t`=3 | `k`=0
7  `t`=3 | `k`=1
8  `t`=4 | `k`=0
9  `t`=4 | `k`=1
~~~

## 1 logits（经 `router_w` 后） -> `expert_ids`

~~~
0  L[t,0]=10 | L[t,1]=30 | L[t,2]=20 | slot0->e=1 | slot1->e=2
1  L[t,0]=15 | L[t,1]=5 | L[t,2]=25 | slot0->e=2 | slot1->e=0
2  L[t,0]=8 | L[t,1]=40 | L[t,2]=12 | slot0->e=1 | slot1->e=2
3  L[t,0]=3 | L[t,1]=9 | L[t,2]=7 | slot0->e=1 | slot1->e=2
4  L[t,0]=11 | L[t,1]=2 | L[t,2]=18 | slot0->e=2 | slot1->e=0
~~~


expert_ids （ `e`）:  1 2 2 0 1 2 1 2 2 0

~~~
0  `t`=0 | `k`=0 | `e`=1
1  `t`=0 | `k`=1 | `e`=2
2  `t`=1 | `k`=0 | `e`=2
3  `t`=1 | `k`=1 | `e`=0
4  `t`=2 | `k`=0 | `e`=1
5  `t`=2 | `k`=1 | `e`=2
6  `t`=3 | `k`=0 | `e`=1
7  `t`=3 | `k`=1 | `e`=2
8  `t`=4 | `k`=0 | `e`=2
9  `t`=4 | `k`=1 | `e`=0
~~~

## 2 `x[t]`：原始 token 表示（未经 `router_w`）

`permuted_x[pos,:] := x[t,:]`

~~~
0  ->  [0.01, 0.02, 0.03, 0.04]
1  ->  [1.01, 1.02, 1.03, 1.04]
2  ->  [2.01, 2.02, 2.03, 2.04]
3  ->  [3.01, 3.02, 3.03, 3.04]
4  ->  [4.01, 4.02, 4.03, 4.04]
~~~

~~~sh
行优先 x（每括号 4 float）： [x0][x1][x2][x3][x4]
~~~

## 3 设备侧缓冲（示意）

~~~
`expert_ids`    ->  10
`x`             ->  5×4
`counts`        ->  3
`offsets`       ->  4
`next_slot`     ->  3
`permuted_x`    ->  10×4
`source_token`  ->  10
`source_k`      ->  10
~~~

## 4 hist -> `counts`

~~~
init  `e`=— | `counts[0]`=0 | `counts[1]`=0 | `counts[2]`=0
0  `e`=1 | `counts[0]`=0 | `counts[1]`=1 | `counts[2]`=0
1  `e`=2 | `counts[0]`=0 | `counts[1]`=1 | `counts[2]`=1
2  `e`=2 | `counts[0]`=0 | `counts[1]`=1 | `counts[2]`=2
3  `e`=0 | `counts[0]`=1 | `counts[1]`=1 | `counts[2]`=2
4  `e`=1 | `counts[0]`=1 | `counts[1]`=2 | `counts[2]`=2
5  `e`=2 | `counts[0]`=1 | `counts[1]`=2 | `counts[2]`=3
6  `e`=1 | `counts[0]`=1 | `counts[1]`=3 | `counts[2]`=3
7  `e`=2 | `counts[0]`=1 | `counts[1]`=3 | `counts[2]`=4
8  `e`=2 | `counts[0]`=1 | `counts[1]`=3 | `counts[2]`=5
9  `e`=0 | `counts[0]`=2 | `counts[1]`=3 | `counts[2]`=5
~~~

~~~
counts  ->  [2, 3, 5]
~~~

## 5 exclusive-prefix -> `offsets`

~~~
0  `sum`（写前）=0 | `offsets[e]`=0 | `sum`（写后 += `counts[e]`）=2
1  `sum`（写前）=2 | `offsets[e]`=2 | `sum`（写后 += `counts[e]`）=5
2  `sum`（写前）=5 | `offsets[e]`=5 | `sum`（写后 += `counts[e]`）=10
~~~

~~~
10  ->  `[0, 2, 5, 10]`
~~~

## 5′ `offsets` ↔ 行区间

~~~
0  行 `[lo, hi)`=[0, 2) | `hi-lo`=2
1  行 `[lo, hi)`=[2, 5) | `hi-lo`=3
2  行 `[lo, hi)`=[5, 10) | `hi-lo`=5
~~~

## 6 `init_next`

~~~
next_slot init  ->  [0, 2, 5]
~~~

`ns ≡ [ns0, ns1, ns2]`

## 7 scatter（`ns_before`->`pos`->写行->`ns_after`）

~~~
0  `(t,k)`=(0,0) | `e`=1 | `ns_before`=[0,2,5] | `pos`=2 | ← `x[t]`=row2 | `ns_after`=[0,3,5]
1  `(t,k)`=(0,1) | `e`=2 | `ns_before`=[0,3,5] | `pos`=5 | ← `x[t]`=row5 | `ns_after`=[0,3,6]
2  `(t,k)`=(1,0) | `e`=2 | `ns_before`=[0,3,6] | `pos`=6 | ← `x[t]`=row6 | `ns_after`=[0,3,7]
3  `(t,k)`=(1,1) | `e`=0 | `ns_before`=[0,3,7] | `pos`=0 | ← `x[t]`=row0 | `ns_after`=[1,3,7]
4  `(t,k)`=(2,0) | `e`=1 | `ns_before`=[1,3,7] | `pos`=3 | ← `x[t]`=row3 | `ns_after`=[1,4,7]
5  `(t,k)`=(2,1) | `e`=2 | `ns_before`=[1,4,7] | `pos`=7 | ← `x[t]`=row7 | `ns_after`=[1,4,8]
6  `(t,k)`=(3,0) | `e`=1 | `ns_before`=[1,4,8] | `pos`=4 | ← `x[t]`=row4 | `ns_after`=[1,5,8]
7  `(t,k)`=(3,1) | `e`=2 | `ns_before`=[1,5,8] | `pos`=8 | ← `x[t]`=row8 | `ns_after`=[1,5,9]
8  `(t,k)`=(4,0) | `e`=2 | `ns_before`=[1,5,9] | `pos`=9 | ← `x[t]`=row9 | `ns_after`=[1,5,10]
9  `(t,k)`=(4,1) | `e`=0 | `ns_before`=[1,5,10] | `pos`=1 | ← `x[t]`=row1 | `ns_after`=[2,5,10]
~~~

~~~
2, 5, 10  ->  `[2,5,10]`
~~~

~~~sh
float 递增 -> | e0 : 8 float | e1 : 12 float | e2 : 20 float |
行 pos       [0 , 2 )      [2 , 5 )        [5 , 10 )
             ×4=float 偏移  ×4              ×4
~~~

## 8 `source_token` / `source_k`

~~~
0  `source_token[pos]`=1 | `source_k[pos]`=1
1  `source_token[pos]`=4 | `source_k[pos]`=1
2  `source_token[pos]`=0 | `source_k[pos]`=0
3  `source_token[pos]`=2 | `source_k[pos]`=0
4  `source_token[pos]`=3 | `source_k[pos]`=0
5  `source_token[pos]`=0 | `source_k[pos]`=1
6  `source_token[pos]`=1 | `source_k[pos]`=0
7  `source_token[pos]`=2 | `source_k[pos]`=1
8  `source_token[pos]`=3 | `source_k[pos]`=1
9  `source_token[pos]`=4 | `source_k[pos]`=0
~~~

## 9 recap：按 expert

~~~
0  `r` 集合=3,9 | 行 `[lo, hi)`=[0,2) | 本例 `pos` 顺序写入的 `(t)`=1->4
1  `r` 集合=0,4,6 | 行 `[lo, hi)`=[2,5) | 本例 `pos` 顺序写入的 `(t)`=0->2->3
2  `r` 集合=1,2,5,7,8 | 行 `[lo, hi)`=[5,10) | 本例 `pos` 顺序写入的 `(t)`=0->1->2->3->4
~~~


## 10 dispatch 输出

供 grouped expert GEMM、`expert_offsets` 与 combine 对照。

计数与分段：

~~~
`counts`  ->  `[2, 3, 5]`
`offsets`（常为 host 侧的 `expert_offsets`）  ->  `[0, 2, 5, 10]`
~~~

专家 `e` 独占 `permuted_x` 行区间 **`[offsets[e], offsets[e+1])`**（可与 §9 对照）。

`permuted_x`（10×4）：

~~~
0  专家 `e`=0 | `x[t]`=`x[1]` | 向量（同 §2）=[1.01, 1.02, 1.03, 1.04]
1  专家 `e`=0 | `x[t]`=`x[4]` | 向量（同 §2）=[4.01, 4.02, 4.03, 4.04]
2  专家 `e`=1 | `x[t]`=`x[0]` | 向量（同 §2）=[0.01, 0.02, 0.03, 0.04]
3  专家 `e`=1 | `x[t]`=`x[2]` | 向量（同 §2）=[2.01, 2.02, 2.03, 2.04]
4  专家 `e`=1 | `x[t]`=`x[3]` | 向量（同 §2）=[3.01, 3.02, 3.03, 3.04]
5  专家 `e`=2 | `x[t]`=`x[0]` | 向量（同 §2）=[0.01, 0.02, 0.03, 0.04]
6  专家 `e`=2 | `x[t]`=`x[1]` | 向量（同 §2）=[1.01, 1.02, 1.03, 1.04]
7  专家 `e`=2 | `x[t]`=`x[2]` | 向量（同 §2）=[2.01, 2.02, 2.03, 2.04]
8  专家 `e`=2 | `x[t]`=`x[3]` | 向量（同 §2）=[3.01, 3.02, 3.03, 3.04]
9  专家 `e`=2 | `x[t]`=`x[4]` | 向量（同 §2）=[4.01, 4.02, 4.03, 4.04]
~~~


这里已经体现出了每一个专家分配的 token 是连续存储的。
