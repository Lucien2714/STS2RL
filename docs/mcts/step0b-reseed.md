# Step 0b：reseed（重设隐藏信息）

*2026-10-03*

- **做了什么**（STS2Simulator `42e330e`）：
  - 新增 `Sts2Sim.Engine/Determinizer.cs`，重设三样东西：
    1. 跑局级的战斗 RNG 流（`RunRngSet` 的 Shuffle、CombatCardGeneration、CombatPotionGeneration、CombatCardSelection、CombatEnergyCosts、CombatTargets、MonsterAi、Niche、CombatOrbs），用游戏自带的 `MockRng` 逐条替换；
    2. 每个活着的怪物自己的 RNG（`monster.Rng`，部分招式会用到）；
    3. 抽牌堆的顺序（`CardPile._cards`）。只换种子不够，已经洗好的顺序还在。
  - 每样东西的种子由"给定种子 + 组件名"确定性地派生。不改动任何可见状态（手牌、牌堆内容、HP、格挡、能力、已经显示的意图），也不碰宏观流（卡牌奖励、商店、地图）。
  - `POST /sim/reseed {seed}`：战斗中途重设；有阻塞决策时拒绝。replay 模式会把它写进请求日志，恢复时一起重放。
  - reset 新增可选的 `reseed`：恢复快照后、进入房间前替换战斗流（开局洗牌也随之改变），战斗开始后再重设怪物的 RNG。同一个快照不再每次都是同一场战斗。
  - STS2RL：`STS2Client.sim_reseed`，`sim_reset(..., reseed=)`，以及对应的单元测试。
  - 新增诊断 `/sim/debug/hidden`（各条战斗流的种子和计数、怪物 RNG、抽牌顺序）和检验脚本 `STS2Simulator/tests/sts2rl/reseed_check.py`。
- **测试**（`reseed_check.py`，每个快照检查五项：不可见、生效、可复现、有变化、reset 时 reseed）：
  - clone 模式 50 个快照：**50/50 通过**。
  - replay 模式 30 个快照：**30/30 通过**。
  - "不可见"的标准是 reseed 前后状态 JSON 完全相同，并且 STS2RL `GameTokenizer` 的输出逐元素相等。
- **检查发现的问题**：
  1. **真实的信息泄露**：抽牌堆在 JSON 里按稀有度和卡牌 id 排序显示，同 id 但升级等级或附魔不同的副本（例如升级过的打击和普通打击）在排序时并列，而不稳定排序会按抽牌顺序排列它们，所以 JSON 会随隐藏的抽牌顺序变化。tokenizer 的输出这次恰好没受影响，但这仍然是泄露。→ `StateBuilder` 在稀有度和 id 之后，用每张卡显示出来的完整 JSON 做最后的排序键，排列只取决于牌堆内容。
  2. 检验脚本的误报：抽牌堆正好等于一手牌的张数时，无论顺序如何都会全部抽到，"不同种子抽到不同手牌"不成立；只有 2–3 张牌时，单次 reseed 有不小的概率顺序不变。→ 只在牌堆大于一手牌时要求手牌有变化；顺序变化改为在多次 reseed 中至少出现一次。
  3. **模拟器的资源泄漏（修复前就存在，被快照放大）**：回归时恢复耗时从 2.5 ms 涨到 83 ms。长时间运行的模拟器（1331 次 reset 之后）从根能到达 248 万个对象、1331 个 `RunState`。用新加的 `/sim/debug/paths` 追到引用链：`NetCombatCardDb._subscriptions` 里每一项持有一个牌堆，通过卡牌一路连到所属的那局。游戏只在 `CombatEnded` 时取消这些订阅，而战斗中途 reset 时 `RunManager.CleanUp` 不会触发这个事件，于是每次 reset 都把那一局留在单例上。快照会记录根能到达的一切，所以越跑越慢。→ `TearDown` 在战斗中先调用数据库自己的结束战斗处理。修复后 30 次 reset（含快照、恢复、reseed）后只剩 1 个 `RunState`、约 3,680 个对象。
  4. 新进程里的偶发"隐藏状态不一致"（只在某个内容第一次出现时）：用 `branch_check.py --dump-on-mismatch` 在新启动的模拟器上抓到 diff，只有 5 行。某个遗物惰性缓存的悬浮提示里，一张卡的标题本地化字符串（`LocString`），第一遍是独立对象，重放时和另一张卡共享同一个对象。原因是规范模型的文本是惰性创建的（规范模型共享、不在快照里），之后复制出来的卡会共享它。内容完全相同（同一张表的同一个条目），只是对象别名不同，不影响玩法。→ 摘要和 dump 对 `LocString` 按"表 + 条目"比较，不再按对象身份比较。
- **最终测试**：
  - `reseed_check.py`：clone 50/50，replay 30/30。
  - `branch_check.py` 在两个新启动的模拟器上：300 条 + 1000 条路线全部通过，没有产生任何 diff；恢复中位数约 **1.3 ms**。
  - 契约回归 200 局：0 个动作错误、0 个模拟器故障。
  - STS2RL `pytest` 614 passed，`ruff` 通过；模拟器侧我写的两个检验脚本也通过 ruff（`sim_harness.py`、`test_reward_discard.py` 里原有的 4 个 ruff 问题不是这次引入的，没有改）。
- **已知的近似**：被效果放到抽牌堆顶、玩家其实知道的牌，也会被一起打乱。追踪玩家知道哪些位置需要在每个"放牌"效果上挂钩子，先不做，等搜索显示这一点重要时再处理（写在 `Determinizer` 的注释里）。
- **闸门**：通过。
- **下一步**：Step 1，MCTS 搜索器（`STS2RL/src/sts2rl/search/`）。
