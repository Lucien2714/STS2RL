# Step 0-pre：重放分支（保底方案）

*2026-10-02*

- **做了什么**：
  - 模拟器（STS2Simulator `0c9040b`）：新增 `Sts2Sim.Screens/ReplayBranches.cs`，记录自 reset 以来所有可能改变对局的请求（包括被拒绝的动作）。分支点 = (reset spec, 请求日志)，恢复 = 用同一个 spec 重新 reset，再按顺序重放日志。新增接口 `POST /sim/snapshot`、`POST /sim/restore {id}`、`DELETE /sim/snapshot/{id}`，分支点可以反复恢复。
  - STS2RL：`STS2Client` 新增 `sim_snapshot` / `sim_restore` / `sim_release`，以及对应的单元测试。
  - 新增端到端检验脚本 `STS2Simulator/tests/sts2rl/branch_check.py`：用 STS2RL 自己的 `GameEnv` 和 `LegalActionProvider` 驱动模拟器，从 boss 快照开局，检查确定性、隔离性和耗时。
- **测试**：
  - `branch_check.py --count 20`（前缀 8 步、之后 40 步、每个快照 3 次绕路）→ 20 个快照全部通过：恢复后状态与分支点完全一致，重放的路线每一步状态哈希都一致，绕路后再恢复没有残留。
  - 同样的检验用前缀 20 步、40 步、之后 60 步，`--rng-seed 1` → 全部通过。
  - `uv run python -m pytest` → 613 passed；`uv run ruff check src tests` → 通过。
  - 模拟器契约回归 `sim_harness.py --contract-episodes 200 --train-episodes 0` → 契约阶段 200 局：0 个动作错误、0 个模拟器故障、0 局超步数上限。
- **检查发现的问题**：
  - 随机策略在 boss 战里死得很快，检验路线偏短。→ 检验脚本的随机策略改成只有 15% 的概率结束回合，路线变长。
  - 契约回归的 parity 阶段报 FAIL：模拟器产生了 3 个词表里没有的 id（`SPEED_POTION_POWER` ×26、`FLEX_POTION_POWER` ×25、休息选项 `HATCH` ×1）。这是 STS2RL 内置词表缺了这几个 id，和这次修改无关（这次只在 API 层加了日志，没有改动状态的构造）。→ 不在这个功能里处理，记为待办。
- **结果**：
  - 恢复耗时约 **1.8 ms + 每条日志 0.12 ms**（日志 0–19 条：1.8 ms；20–39 条：4.1 ms；40–59 条：6.8 ms）。一场 boss 战 40–120 个动作，恢复约 6–16 ms。
  - Python 端每走一步（HTTP + 解析状态 + 枚举合法动作）中位数 1.3–2.5 ms，p90 2–3.7 ms。
  - 由此估算两回合树的一次模拟（10–30 个动作）约 20–60 ms，其中恢复只占约 20%，主要开销在 Python 端走步。
- **闸门**：通过。恢复耗时落在计划的"5–50 ms 可以用"区间的低端。
- **下一步**：深拷贝快照（0a）原本是为了把恢复从预计的 40–50 ms 降下来；实测重放只要 6–16 ms，而且不是瓶颈，所以 0a 的收益有限，需要和你确认是否继续做。0b（reseed）无论如何都需要，先做 0b。
