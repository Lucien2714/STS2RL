# MCTS 战斗搜索：进度日志

计划见 `C:\Users\Lucien2714\.claude\plans\progress-temporal-honey.md`。STS2RL 的修改在 `feat/mcts-combat` 分支，模拟器的修改在 STS2Simulator 的 `feat/state-snapshot` 分支。每个子步骤完成后都按"测试 → 检查 → 记录 → 提交"走完再进入下一步。

---

## 2026-10-02 Step 0-pre：重放分支（保底方案）

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

---

## 2026-10-03 Step 0a：内存快照（原地恢复）

- **做了什么**（STS2Simulator `d9bad0b`）：
  - 新增 `Sts2Sim.Engine/Snapshots/StateSnapshot.cs`。快照不新建对象，而是记录每个可变对象的字段值（`MemberwiseClone`），恢复时用按类型编译的 IL 把字段写回**同一批对象**。对象身份不变，所以挂起的 async 续体、按身份哈希的字典（`NetCombatCardDb` 以卡牌对象为键）、事件订阅都保持有效；快照之后新建的对象在恢复后自然变成垃圾。
  - 快照的根：`RunManager`、`CombatManager`、`NetCombatCardDb`、存档的全局进度 `SaveManager.Instance.Progress`、`SimSession`、卡牌选择器栈，以及 4 个会被重新赋值的静态字段（Chaotic RNG、两个选择器、最近一次选卡界面）。规范模型（`IsCanonical`）、反射元数据、模拟器自己的线程设施共享，不记录。
  - 在有阻塞决策（选卡等界面）时拒绝做快照；恢复前先取消当前挂起的决策，让游戏线程退出（和 reset 的 `TearDown` 同样的做法）。
  - 服务器新增 `--snapshot-mode clone|replay`，**clone 成为默认**，replay 保留作为对照。
  - 新增诊断：`/sim/debug/census`（对象图统计和"不能直接复制"的类型）、`/sim/debug/statics`（每个静态字段的摘要）、`/sim/debug/digest`（整张对象图的内容哈希）、`/sim/debug/dump`（按"路径 = 值"导出对象图，用于逐字段 diff）。`GameLoop` 增加看门狗：一次 settle 超过 100 万个续体就报错，而不是静默空转。
  - `branch_check.py` 增强：每个分支点跑多条随机路线，每条走两遍逐步比较；在 clone 模式下额外比较整张对象图的摘要（包括抽牌顺序、RNG 计数器等隐藏状态）。
- **测试**：
  - clone 模式 `branch_check.py --count 50 --lines 20 --detours 3`：**1000 条随机路线全部通过**，每一步的可见状态和最后的隐藏状态摘要都一致，0 个故障。
  - 1 万次"恢复 + 3 步随机动作"：内存在 190–215 MB 之间随 GC 波动，**没有增长**。
  - replay 模式 `branch_check.py --count 20 --lines 5`：100 条路线全部通过（可见状态）。
  - 默认参数启动的冒烟测试：模式为 clone，10 个快照 50 条路线全部通过。
  - 契约回归 `sim_harness.py --contract-episodes 200`：0 个动作错误、0 个模拟器故障（parity 阶段的 3 个词表缺失同 0-pre，与本次无关）。
- **检查发现的问题**（按发现顺序）：
  1. `Nullable<T>` 数组里没有值的元素装箱后是 null，遍历时空引用崩溃。→ 跳过 null 元素。
  2. 恢复后 `end_turn` 卡死 15 秒。用 `dotnet-stack` 抓到游戏线程卡在 `Decision<RewardChoice>.Ask`：绕路的随机路线赢了 boss，进入奖励界面，游戏线程阻塞在这个决策里；恢复把 `_pending` 写回成"没有决策"，但线程还停在另一条时间线的决策里。→ 恢复前先取消挂起的决策并等待游戏线程退出。同时发现恢复会把 `Fault` 也写回"无故障"，掩盖了问题。→ 已故障的会话拒绝恢复。
  3. 装箱的结构体原本被当作不可变共享，没有记录也没有遍历。→ 改为像普通对象一样记录，写回时通过 `unbox` 原地修改。（这一项不是问题 2 的原因，但属于覆盖不全。）
  4. 3/50 个快照"每一步可见状态都一致，但最后隐藏状态不一致"。用 dump 逐字段 diff，发现差在 `Player.DiscoveredRelics/Potions`：第一次拿到某个遗物时，游戏会写入全局存档进度并记在玩家身上；重放时全局进度里已经"发现过"了，就不再记。这些字段只被结算界面和图鉴读取，不影响玩法。→ 把全局进度对象加入快照的根，恢复时一起回滚。
  5. replay 模式在新增的隐藏状态比较下全部"失败"：reset 不会清掉单例上跨局累积的簿记（集合的版本计数、惰性缓存、全局进度），dump 确认都不影响玩法。→ 隐藏状态比较只在 clone 模式下做（只有 clone 声称隐藏状态完全一致）。
- **结果**：
  - 一次快照覆盖约 2.1 万个对象，**捕获约 5 ms**；**恢复约 2.5 ms**（含 HTTP 和构造状态 JSON；replay 方式是 6–16 ms，且随战斗长度增长）。
  - 恢复后每走一步 Python 端约 1.2–1.6 ms。
- **闸门**：通过。正确性、隔离性、内存三项检验全部满足，恢复耗时 ≤ 5 ms，属于计划里的"理想"档。
- **备注**：调试时以用户级全局工具的方式安装了 `dotnet-stack`（`dotnet tool uninstall -g dotnet-stack` 可移除）。
- **下一步**：0b，reseed。

---

## 2026-10-03 Step 0b：reseed（重设隐藏信息）

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
