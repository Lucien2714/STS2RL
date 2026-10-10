# 通关记录

每次通关追加一条。检查点和这一局的战斗记录保存在 `runs/wins/<seed>/`（`runs/` 不进 git，只在本机）；`info.json` 里有指标、检查点的 sha256 和三场 boss 战。保存用 `runs/tools/save_win.py <run> <局号>`：它复制覆盖这一局的检查点（开始前最后一个、局中的、结束后第一个），按局的步数（允许差 2 步）和派发顺序（lane 内的局序号）找出这一局的战斗记录。`runs/step2w_win_saver.sh` 在训练中每 2 分钟自动运行它。同一个 seed 再次通关时保存到 `runs/wins/<seed>-ep<局号>/`。

通关的判断：指标里没有胜利字段，打赢的 boss 数 =（奖励 −（楼层 − 1）+ 0.01 × 步数）/ 10，通关是 3。这些 run 每一幕只有一个 boss，所以 3 个 boss 奖励只能来自通关。

| 日期 | seed | run | 局 | boss | 进入 boss 战时 HP | 检查点 |
|---|---|---|---|---|---|---|
| 2026-10-09 | `JEQSXL4XVT` | `runs/step2w-plain10` | 2210（奖励 71.23，楼层 48，577 步） | Soul Fysh → The Insatiable → Aeonglass | 69/82 → 82/82 → 80/82 | `update_000742`（开始前）、`update_000748`（局中）、`update_000754`（结束后） |
| 2026-10-09 | `3EH64E978B` | `runs/step2w-plain10` | 2285（周期 16） | Lagavulin Matriarch → The Insatiable → Aeonglass | 87/87 → 77/87 → 88/101 | `update_000772`（开始前）、`update_000778`（局中）、`update_000784`（结束后） |
| 2026-10-09 | `3EH64E978B`（第二次） | `runs/step2w-plain10` | 2734（周期 19，奖励 72.37，463 步） | Lagavulin Matriarch → The Insatiable → Aeonglass | 87/87 → 40/87 → 85/101 | `update_000934`（开始前）、`update_000940`（局中）、`update_000946`（结束后）；保存在 `runs/wins/3EH64E978B-ep2734/` |
| 2026-10-09 | `4UEVB4R020` | `runs/step2w-plain10` | 2746（周期 19，奖励 71.15，585 步） | Ceremonial Beast → Knowledge Demon → Aeonglass | 82/85 → 85/85 → 74/86 | `update_000934`（开始前）、`update_000940`、`update_000946`（局中）、`update_000952`（结束后） |
| 2026-10-10 | `4UEVB4R020`（第二次） | `runs/step2y-refkl-hp` | 2447（周期 17，奖励 71.96，504 步） | Ceremonial Beast → Knowledge Demon → Aeonglass | 80/80 → 80/80 → 80/80 | `update_000703`（开始前）、`update_000709`（局中）、`update_000715`（结束后）；保存在 `runs/wins/4UEVB4R020-step2y-refkl-hp-ep2447/` |
| 2026-10-10 | `YMQ4BC2NH5` | `runs/step2y-refkl-hp` | 2578（周期 18，奖励 70.45，655 步） | Kin Follower ×2 + Kin Priest → The Insatiable → Torch Head Amalgam + Queen | 65/80 → 80/80 → 100/100 | `update_000745`（开始前）、`update_000751`、`update_000757`（局中）、`update_000763`（结束后） |

## 2026-10-09：第一次通关（`JEQSXL4XVT`）

- 训练：Step 2-w 普通 PPO（没有探索项和参考 KL），从 Step 2-v 末尾继续；战斗由 MCTS 决定（普通和精英战 50 次模拟，boss 战 200 次，树深 2 回合，权重 `act1_boss_step1b.json`）。
- 这一局从约第 744 次更新开始，到第 751 次更新结束，期间策略更新了 8 次，所以没有一个检查点完全对应这一局。
- 搜索记录里的 run id 是 `s2124-lane4-5`：最后一个战斗决策是第 576 步，这一局共 577 步。最后一个决策时玩家 21 HP、Aeonglass 57 HP，下一步结束了战斗。
- 更正（2026-10-09 03:40）：这一条最初写的"进入 boss 战时 HP 53/82 → 27/82 → 21/82"取的是每场 boss 战最后一个决策的 HP，不是进入时的 HP；进入时是 69/82 → 82/82 → 80/82。
- 这个 seed 在本 run 的 15 次对局（每个周期一次）：楼层 17、33、33、17、17、21、17、24、48、17、17、33、17、17、48（通关）。第 1297 局（周期 9）死在第 3 幕 boss。

## 2026-10-09：第二次通关（`3EH64E978B`）

- 周期 16 的第 2285 局，搜索记录 run id `s2124-lane1-18`，这一局期间策略从第 775 次更新变到第 781 次更新。
- 三场 boss 战都是满血或接近满血进入：Lagavulin Matriarch 87/87，The Insatiable 77/87，Aeonglass 88/101。
- 两次通关的第 2、3 幕 boss 相同：The Insatiable 和 Aeonglass。
- 这个 seed 在本 run 的 16 次对局：楼层 17、7、17、28、25、28、17、23、17、17、17、17、17、17、17、48（通关）。前 15 次有 10 次死在第 1 幕 boss，以前最高只到第 28 层。

## 2026-10-09：第三、四次通关（周期 19）

- 第 2734 局：`3EH64E978B` 第二次通关（第一次是第 2285 局）。这一次进入第 2 幕 boss 时只有 40/87 HP，仍然打赢了。
- 第 2746 局：`4UEVB4R020`。这个 seed 以前两次死在第 3 幕 boss（第 196 局、第 1847 局）。
- 四次通关的第 3 幕 boss 都是 Aeonglass。
- 找战斗记录的规则改过两次：这两局的最后一个战斗决策和局的步数差 1（不是 2210、2285 那样的倒数第二步），而且按步数有两个候选（`s2124-lane8-12` 和 `s2124-lane5-59`），按派发顺序选出 `lane5-59`。

## 2026-10-10：第五次通关（Step 2-y 第一次，`4UEVB4R020`）

- 训练：Step 2-y（`runs/step2y-refkl-hp`），从 Step 2-w 的 `update_001035` 开始，营火参考 KL β 0.3，只在 HP ≥ 50% 的营火界面生效（`--reference-kl-min-hp rest_site=0.5`），地图 β 0.03。
- 周期 17 的第 2447 局，搜索记录 run id `s0-lane5-231`，这一局期间策略从第 707 次更新变到第 713 次更新。
- 三场 boss 战都是满血进入（80/80）。boss 和 Step 2-w 第 2746 局的通关完全相同：Ceremonial Beast → Knowledge Demon → Aeonglass。
- 这个 seed 在本 run 的 16 次对局：楼层 24、38、33、33、9、48、37、17、17、28、28、33、24、17、48、48（通关）。第 953 局和第 2298 局死在第 3 幕 boss。
- 背景：这一局发生在第 10–14 周期的崩溃（选牌奖励几乎总是跳过，平均楼层约 15–16）之后，第 15–17 周期选牌恢复到拿卡约 0.65，楼层回到约 21–23。

## 2026-10-10：第六次通关（Step 2-y 第二次，`YMQ4BC2NH5`）

- 周期 18 的第 2578 局，搜索记录 run id `s0-lane7-255`，这一局期间策略从第 750 次更新变到第 758 次更新，共 655 步、24 分钟。
- 第一个通关的新 seed，也是第一次第 3 幕 boss 不是 Aeonglass 的通关：Kin Follower ×2 + Kin Priest（65/80 进入）→ The Insatiable（80/80）→ Torch Head Amalgam + Queen（100/100）。
- 这个 seed 在本 run 的 17 次对局：楼层 17、14、13、17、17、23、17、17、17、12、17、15、12、11、17、48、48（通关）。前 15 次最高第 23 层，第 2412 局第一次到第 3 幕 boss 并输掉，下一次（本局）通关。
