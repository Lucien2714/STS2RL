# Step 2-i：精英和 boss 战用 200 次模拟（`runs/step2i-mcts200`）

*2026-10-05*

## 做了什么

`feae10e`：每种房间可以有自己的模拟次数。

- `TrainingConfig` 新增 `search_elite_simulations` / `search_boss_simulations`（`--search-elite-simulations`、`--search-boss-simulations`，`sts2rl-eval` 也支持），`search_room_simulations()` 给出 `{"elite": n, "boss": n}`；
- `SearchCombatAgent` 记下当前房间，`CombatSearch.search(env, state, simulations=...)` 用这个房间的预算（必须 ≥ 1），没有设置的房间用默认值；
- resume 时这两个字段和其他搜索设置一样不能改。

run：从 boss run 的 `update_000192` 用 `--init-from` 开始，普通战斗 50 次模拟，精英和 boss 战 200 次。

## 结果（178 局，55 次更新，10-05 09:11 ~ 12:50）

- 第 1 周期平均 floor 20.3，114 局到第 17 层。
- 和 boss run 同一批 seed 配对：**+0.8 层，p = 0.67**，不显著。
- 每局的时间是 50 次模拟时的 **2.2 倍**。

按时间算不划算，所以停止，回到 50 次。停止的另一个原因是同时发现了 Waterfall Giant 的评估错误（Step 2-j），先修它。

## 快照测试（固定的 429 个 boss 快照）

修复评估之后，在同样的 173 场战斗上比较：200 次模拟 35%，50 次 29%（测试还在进行，最终数字写进 Step 2-k）。
