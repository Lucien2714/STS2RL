# MCTS 战斗搜索：每一步的记录

计划：`C:\Users\Lucien2714\.claude\plans\progress-temporal-honey.md`。STS2RL 的修改在 `feat/mcts-combat` 分支，模拟器的修改在 STS2Simulator 的 `feat/state-snapshot` 分支。

每个子步骤完成后都按"测试 → 检查 → 记录 → 提交"走完，并在这个目录里单独保存一份记录。

| 步骤 | 记录 | 日期 |
|---|---|---|
| Step 0-pre：重放分支（保底方案） | [step0-pre-replay-branches.md](step0-pre-replay-branches.md) | 2026-10-02 |
| Step 0a：内存快照（原地恢复） | [step0a-in-place-snapshots.md](step0a-in-place-snapshots.md) | 2026-10-03 |
| Step 0b：reseed（重设隐藏信息） | [step0b-reseed.md](step0b-reseed.md) | 2026-10-03 |
| 词表更新（插入的任务）：以游戏 v0.107.1 为准 | [vocabulary-update.md](vocabulary-update.md) | 2026-10-03 |
| Step 1-a：MCTS 搜索器 | [step1a-search.md](step1a-search.md) | 2026-10-03 |
| Step 1-b：拟合叶子评估 | [step1b-leaf-evaluator.md](step1b-leaf-evaluator.md) | 2026-10-03 |
| Step 1-c：选搜索深度 | [step1c-depth.md](step1c-depth.md) | 2026-10-03 |
| Step 1-d：全量实验 | [step1d-full-experiment.md](step1d-full-experiment.md) | 2026-10-03 |
| Step 2-a/b/c：分层 agent 与冒烟测试 | [step2abc-layered-agent.md](step2abc-layered-agent.md) | 2026-10-03 |
| Step 2-d：MCTS 打战斗的 PPO 训练与评估 | [step2d-training-run.md](step2d-training-run.md) | 2026-10-04 |
| Step 3-a：蒸馏战斗 actor | [step3a-distillation.md](step3a-distillation.md) | 2026-10-04 |
| Step 2-e：稳定宏观训练（KL 上限、战斗移出 rollout） | [step2e-stabilisation.md](step2e-stabilisation.md) | 2026-10-04 |
