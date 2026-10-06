# Step 2-j：Waterfall Giant 的评估错误，以及输赢之外的诊断（`runs/step2j-fixed`）

*2026-10-05*

## Waterfall Giant 的评估错误（`70068df`）

Steam Eruption："When killed, deals damage at the end of your next turn." Giant 被打死后会在场上留一个回合，HP 显示为 999,999,999 / 999,999,999。叶子评估把它读成一个满血的 boss：`enemy_hp_ratio` 从约 0.04 跳到 1.0（权重 −4.0），race 特征看到的是一个打不死的敌人。**搜索因此避开了最后一击**：298 场输掉的 Giant 战中，176 场结束时 Giant 的 HP 低于 15%；Giant 的胜率 10%，其他第 1 幕 boss 是 50–77%。

修复：max HP ≥ 100,000,000 的敌人是"正在死"，它的 HP 不计入敌人总量，它的 Steam Eruption 计入来袭伤害（搜索仍然会为爆炸格挡）。测试：被打死的 Giant 的 `enemy_hp_ratio` 是 0；在同样威胁（eruption = 10）下，打死比"几乎打死"分数高；爆炸能致死时 `lethal_incoming` 为 1。

快照测试（MCTS-50，66 个 Giant 快照）：旧评估 1/66，修复后 3/66，其中 29 场打死了 Giant（26 场死于爆炸）。全部 429 个快照 104 → 106 胜，其他 boss 没有变化。训练中 Giant 的胜率从约 14% 升到 33–50%。

## 输赢之外：HP、回合数和牌组

按指示不只看输赢，而是看掉了多少 HP、打了几回合。用 boss 战的数据做回归（在留出的 run 上算 R²）：

| 目标 | 只用战斗前的 HP 等 | 加上牌组 |
|---|---|---|
| 每回合造成的伤害 | 0.15 | **0.47** |
| 战斗余量（输：boss 剩余 HP；赢：自己剩余 HP） | 0.19 | **0.42** |

**牌组解释了大部分差别**。和人类（17 局录像）在第 1 幕 boss 前比较：

| | 人类 | agent |
|---|---|---|
| 打过第 1 幕 boss | 17/17 | 55% |
| 进入 boss 战时 HP | 77% | 88% |
| 升级的牌 | 3.8 | 0.9 |
| 药水 | 2.5 | 0.7 |
| 牌组大小 | 17.1 | 19.8 |

约 40% 的 agent boss 战开始时还有 150 以上没花的金币。agent 用更多 HP 进 boss 战，但牌组更大、升级更少、药水更少。

## 新 run：`--init-from` 之后的掉落

run：从 Step 2-i 的 `update_000054` 用 `--init-from`，全部 50 次模拟，带上评估修复。

| 周期 | 局 | 平均 floor | 到第 17 层 |
|---|---|---|---|
| 1 | 1–150 | **17.5** | 92 |
| 2 | 151–291 | 19.2 | 99 |

第 1 周期比 boss run 同一批 seed 低约 2 层。原因之一：`--init-from` 只带模型和回报缩放，**Adam 的一阶、二阶矩重新开始**，前几次更新选牌的 KL 约为正常的 3 倍。所以加了 `--init-optimizer`（`ef26810`）：

- `InitialWeights` 带上来源 checkpoint 的优化器状态；
- `CandidatePPOAgent._load_optimizer_moments` 检查参数组数、参数个数，以及每个 `exp_avg` / `exp_avg_sq` 的形状和参数一致，再加载，并把学习率设为新 run 的值；
- `--init-optimizer` 必须和 `--init-from` 一起用；来自另一个模型的矩会被拒绝。

约 290 局时（比 boss run 低 1.2 层）停止，从 boss run 的平台期重新开始（Step 2-k）。
