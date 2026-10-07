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
| Step 2-f：模型迁移工具（surgery、`--init-from`） | [step2f-surgery.md](step2f-surgery.md) | 2026-10-04 |
| Step 2-g：稳定后的宏观训练（600 局） | [step2g-stable-run.md](step2g-stable-run.md) | 2026-10-05 |
| Step 2-h：当前幕 boss 特征（`act_boss`）和 boss run | [step2h-boss-column.md](step2h-boss-column.md) | 2026-10-05 |
| Step 2-i：精英和 boss 战用 200 次模拟 | [step2i-mcts200.md](step2i-mcts200.md) | 2026-10-05 |
| Step 2-j：Waterfall Giant 的评估错误、牌组诊断、`--init-optimizer` | [step2j-giant-fix.md](step2j-giant-fix.md) | 2026-10-05 |
| Step 2-k：从平台期继续训练，以及冻结策略的对照 | [step2k-plateau.md](step2k-plateau.md) | 2026-10-05 |
| Step 2-l：叶子评估里的击杀和死亡召唤（A/B +1.9 层） | [step2l-kill-evaluation.md](step2l-kill-evaluation.md) | 2026-10-05 |
| Step 2-n：修复评估之后的宏观训练（600 局；宝箱不拿遗物） | [step2n-training.md](step2n-training.md) | 2026-10-06 |
| Step 2-o：宝箱必须先拿遗物，再离开 | [step2o-treasure.md](step2o-treasure.md) | 2026-10-06 |
| Step 2-p（修复）：PPO 更新时保留正在打战斗的决策 | [step2p-held-steps.md](step2p-held-steps.md) | 2026-10-06 |
| Step 2-p（性能）：训练进程的 HTTP 开销（urllib3、restore 带 reseed） | [step2p-perf.md](step2p-perf.md) | 2026-10-06 |
| Step 2-p（训练）：修复奖励归属之后，从最好的周期继续训练 450 局 | [step2p-training.md](step2p-training.md) | 2026-10-06 |
| Step 2-q：商店里的 Foul Potion，以及被拒绝后不重复同一动作 | [step2q-foul-potion.md](step2q-foul-potion.md) | 2026-10-06 |
| Step 2-r：在 holdout seed 上评估四个 checkpoint，以及宏观策略的探测 | [step2r-holdout-eval.md](step2r-holdout-eval.md) | 2026-10-06 |
| Step 2-s：定向探索（ε 混合采样；没有恢复塌缩的选项） | [step2s-exploration.md](step2s-exploration.md) | 2026-10-07 |
| Step 2-t / 2-u：向人类克隆策略的 KL 约束（全局 β 失败；按界面 β 把升级概率从 0.01 拉到 0.27，楼层持平） | [step2tu-reference-kl.md](step2tu-reference-kl.md) | 2026-10-07 |
