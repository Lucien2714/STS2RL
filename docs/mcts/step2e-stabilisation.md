# Step 2-e：稳定宏观训练（每次更新的 KL 上限、战斗移出 rollout）

*2026-10-04*

Step 2-d 的结论：宏观策略学到了东西（最好的 checkpoint 比 intent8h 高 2.7 层），但训练不稳定，最后摆回了 intent8h 的水平。这一步加两个改动，都默认关闭，用新参数打开。提交 `f160c3e`。

## 1. 每次更新的 KL 上限（`--target-kl`）

**PPO 不是已经有 KL 限制了吗？** 没有完全限制。PPO 的 clip 只限制"这个 batch 里出现过的 (状态, 动作)"的概率比 r 在 [1−ε, 1+ε] 之外不再产生梯度，它有三个漏洞：

- 一次更新是 4 个 epoch × 8 个 minibatch = 32 次优化，clip 只是让超出范围的样本不再有梯度，不会把策略拉回来，32 步累积下来整体可以走得比 ε 远很多；
- 编码器是共享的。不在 batch 里的状态（比如一局只出现几次的选牌界面）没有任何 clip 保护，其他状态的更新会顺带改变它们的输出；
- 一个动作的 r 被 clip 住了，同一状态下其他候选的概率仍然会被 softmax 重新分配。

Step 2-d 测到的 P(跳过选牌) 从 0.21 一次跳到 1.00 就是第二种情况。

**做法**（和 OpenAI Baselines / Stable-Baselines3 的 `target_kl` 相同）：每个 minibatch 在做优化之前，先用 (r − 1) − log r（KL(旧‖新) 的低方差估计，总是 ≥ 0）算出当前策略离"收集 rollout 的那个策略"有多远；超过 1.5 × target_kl 就结束这次更新剩下的所有 epoch。检查放在 step 之前，所以越过上限的那一步不会执行。

- `PPOConfig.target_kl: float | None = None`，必须为正；`None` 表示不限制（和以前完全一样）。
- 新增指标：`approx_kl`、`optimizer_steps`（这次更新实际做了几步）、`kl_early_stop`（是否提前停止）。
- 写进训练计划，resume 时不能改。

## 2. 战斗移出 rollout（`--search-fights-out-of-rollout`）

以前 MCTS 打的战斗步作为"外部动作"进 rollout：参与 GAE 和 critic，不参与 policy loss。问题是一次 256 步的更新里约 60% 是战斗步，PPO 自己的选择只有约 80 个；而且一次选牌到它影响的 boss 战之间隔了 50–150 步。

打开这个选项后，**整场战斗当作环境的一次转移**：

- `choose_external(state, action, record=False)`：动作照常执行，但不建 rollout 条目；
- 这一步的奖励并进这个 lane 最后一个记录的决策（`_fold`），`next_observation` 也更新为战斗后的状态；战斗中死亡，就把那个决策标成这局的终点；
- **不跨局合并**（CLAUDE.md "Forced steps are folded" 里的教训）：如果最后一个决策已经是上一局的终点，或者这局还没有记录任何决策，奖励先暂存（`carried_reward`），加到这局下一个记录的决策上；这局还没有决策就死了，奖励没有可以归属的决策，丢弃；
- `reset`、`discard_decision`、`abort_lane` 都会清掉暂存状态；每个 lane 各自独立。

效果：256 步的 rollout 全是 PPO 自己的选择（是以前的约 3 倍），选牌到 boss 的距离缩短到 10–30 个决策，critic 只估计战斗之间的状态。

- `TrainingConfig.search_fights_in_rollout: bool = True`（默认保持旧行为）；`SearchCombatAgent(record_fights=…)`。
- 写进训练计划，resume 时不能改。

## 测试

`tests/test_ppo_agent.py`：

- KL：不设上限时一次更新做满 16 步；上限 1e-9 时做完第一步就停（`kl_early_stop` = 1）；`target_kl` 必须为正；
- 合并：战斗奖励并进前一个决策；战斗中死亡会标记终点；终点决策不会被延长，奖励转给下一局的第一个决策；两个 lane 互不影响；只有宏观决策的 rollout 能正常更新。

`tests/test_search_agent.py`：`record_fights=False` 时走不记录的路径。`tests/test_training_cli.py`：两个新参数进入计划，resume 时改动会被拒绝；`tests/test_training_config.py`：旧的计划文件（没有这两个字段）照常加载。

全量回归 681 个通过，`ruff check src tests` 通过。

## 下一步

- Part 3：迁移工具（`--init-from`、词表迁移、schema 列迁移、输出一致性检查），让新的 run 可以从已有的 checkpoint 开始，不用每次从头训练；
- Part 4：从 Step 2-d 最好的 checkpoint（`update_001052`）开始，用这两个选项重新训练宏观策略，和 intent8h 对比。
