# Step 2-f：模型迁移工具（surgery）

*2026-10-04*

## 为什么要做

以前每次更新词表（新卡牌、新 power）或者在 `encoder/schema.py` 里加一列，旧的 checkpoint 就不能用了：词表指纹不一致会被拒绝，列数变了张量形状对不上也加载不了。这两个拒绝都是对的（硬加载的话，旧权重会被当成别的意思用），但结果是每次都要从随机权重重新训练。

OpenAI Five 在十个月里经历了很多次游戏版本更新，用的办法叫 "surgery"：把旧参数改写成新形状，让新模型的输出和旧模型**完全一样**，然后接着训练。这一步就是给 `GameEncoder` 做的这套工具。

## 做了什么

### 1. 迁移规则（`training/surgery.py`）

模型的输入只有两类东西会变，各有一个精确的处理方法：

| 变化 | 处理 | 为什么输出不变 |
|---|---|---|
| 词表加了 token（表是排序的，新 token 插在中间，后面的行全部错位） | 按 token 身份搬行；旧词表没有的 token 用旧的 `<unknown>` 行 | 旧模型看这个 token 时看到的就是 `<unknown>` |
| 新的类别列 | 这一列的 embedding 表全部初始化为 0 | 类别 embedding 是**加**到实体行上的，0 什么都不加 |
| 新的数值列 | 每个数值投影的输入是 `[值..., mask...]`，按字段名搬列，新字段的值列和 mask 列初始化为 0 | 0 权重乘任何输入都是 0 |
| 删掉的列 / 参数 | 丢弃，报告里标出 | 输出会变（如果这一列以前不是 0） |
| 新的实体种类 | 新参数用新的随机初始化，报告里标出 | **不能**保持：新种类的实体会进入 transformer 的序列，权重是多少都会改变注意力 |
| 其他参数形状变了 | 拒绝 | 不是输入的变化，不能自动处理 |

一个"例外是故意的"：以前读成 `<unknown>`、现在能识别的 id（比如 `API_SUFFIXES` 让 `FLEX_POTION_POWER` 能找到 `FLEX_POTION`），新模型会用这个 token 训练过的行，输出会变。这正是更新词表的目的，只能用下面的输出检查来统计。

优化器状态从不迁移：Adam 的动量是按行、按列的，行和列移动以后就没有意义了。

### 2. checkpoint 自带模型说明（`encoder/spec.py`）

旧词表和旧 schema 只存在于旧代码里，旧代码一改就没了。所以现在**每个 checkpoint 都带一份 `model_spec`**：词表的每张表、事件选项、schema 的每一列（约 36 KB）。迁移时直接从 checkpoint 读。

- 这是可选字段，加载 checkpoint 时不读它，所以没有提升 `FORMAT_VERSION`（提升的话现有的所有 checkpoint 都会被拒绝，包括 Part 4 要用的 `update_001052`）。
- 这之前的 checkpoint 没有这份说明，要在训练它的那个版本的代码里用 `sts2rl-surgery spec` 导出一份，迁移时用 `--old-spec` 传入。

### 3. 命令

```bash
# 导出当前代码的模型说明（给没有自带说明的旧 checkpoint 用，在旧代码里运行）
sts2rl-surgery spec old_spec.json

# 把一个 checkpoint 改写成当前代码的形状，写到新的 run 目录（计数器保留，优化器重置）
sts2rl-surgery migrate --from runs/old --to-run runs/old-migrated [--old-spec old_spec.json]

# 输出检查：用记录的决策打分（候选用记录里的，不重新生成），再比较
sts2rl-surgery outputs --from runs/old --dataset data/bc/v6 --out before.json   # 旧代码里
sts2rl-surgery outputs --from runs/old-migrated --dataset data/bc/v6 --out after.json
sts2rl-surgery compare before.json after.json

# 从另一个 run 的整个模型开始一个新的训练计划
sts2rl-train --run-dir runs/next --init-from runs/old-migrated[/checkpoints/update_000100.pt]
```

`scripts/migrate_vocabulary.py` 删掉了，它的功能是 `sts2rl-surgery migrate` 的一部分。

### 4. `sts2rl-train --init-from`

和 `--init-encoder`（从行为克隆的产物开始，只有 actor 训练过）不同，`--init-from` 从另一个 PPO run 的 checkpoint 开始，带上**整个模型**：actor、critic，以及 critic 对应的 `_ReturnScale`（critic 预测的是缩放后的回报，只带 critic 不带缩放比例，比不带 critic 更差）。

- 优化器和计数器重新开始，所以新计划可以改 PPO 设置（学习率、`--target-kl`、`--search-fights-out-of-rollout`）；
- 在建立 run 目录**之前**就加载并检查（词表指纹、编码器尺寸），不兼容的来源不会留下半个 run；词表不一致时报错信息会提示先用 `sts2rl-surgery migrate`；
- 计划里记录的是解析后的 checkpoint **文件**，不是目录（目录的 latest 会变）；
- 不能和 `--resume`、`--init-encoder` 一起用。

新增 `CandidatePPOAgent.initialize_from(encoder_state, return_scale)`。

## 测试

`tests/test_surgery.py`（20 个）。旧模型的代码已经不在了，没法直接构造旧模型来比较，但可以构造它的行为：**一个没有某一列的模型，等于现在的模型把这一列的输入置零**。所以每个测试都是：建一个模型 → 删掉一部分得到"旧"权重 → 迁移 → 和原模型在"去掉那部分输入"的状态上比较输出：

- 不变的模型迁移后和自己完全相同；
- 词表少了 UPPERCUT：其他每一行都回到自己的 token 下，UPPERCUT 用旧的 `<unknown>` 行；输出等于原模型把 UPPERCUT 读成 `<unknown>`；
- 新数值列（gold）：输出等于原模型看不到 gold；并确认 gold 本身确实会影响输出（测试不是空的）；
- 新类别列（卡牌 rarity）：零表，输出等于原模型这一列是 padding；
- 新实体种类（orb）：报告"输出会变"；形状变化在输入之外：拒绝；只有词表、没有 schema 的旧说明遇到列变化：拒绝并提示用 `sts2rl-surgery spec`；
- checkpoint 自带说明；`migrate_checkpoint` 写出的 run 能正常加载，计数器保留、优化器为空、`migration.json` 存在；没有说明的 checkpoint 需要 `--old-spec`；
- `--init-from`：权重和缩放比例都带过来、优化器为空、新学习率生效；编码器尺寸不同或词表不同被拒绝；和 `--resume` 互斥；和 `--init-encoder` 互斥；来源无效时不建 run 目录；计划里记录 checkpoint 文件；
- `outputs` / `compare` 的计数。

全量回归：700 个通过（681 − 1 个删掉的旧脚本测试 + 20 个新测试），`ruff check src tests scripts` 和 `compileall` 通过。

## 用真实数据检查

**1. 重做 intent8h 的词表迁移。** 之前用旧脚本把 intent8h（v0.107.1 词表更新之前训练的）迁移成了 `runs/intent8h-v4`。这次在旧代码里导出说明，用新工具重做一遍：

- 第一次用了词表更新前一个提交 `715b65b` 的代码，**被拒绝**：说明里 intent 的数值输入是 6 列，权重是 10 列。原因是 intent8h 是在 `4e1a30f`（多段攻击意图改成数值）之后训练的，词表指纹相同但 schema 不同。工具按形状发现了这个错误，没有错误地加载。这也说明了为什么要让 checkpoint 自带说明：靠人去找"训练时是哪个版本的代码"很容易找错。
- 改用 `4e1a30f` 的说明：迁移结果和 `intent8h-v4` **逐个张量完全相同**（151/151），回报缩放比例和计数器也相同。

**2. 跨代码版本的输出检查。** 在 `4e1a30f` 的代码里用原始 intent8h 打分，在当前代码里用迁移后的模型打分，数据是 `data/bc/v6` 里每种界面 40 个决策（共 499 个）：

| | 决策 | 完全相同（误差 ≤ 1e-5） | argmax 相同 |
|---|---|---|---|
| 全部 | 499 | 478 | 495 |
| boss | 40 | 30 | 37 |
| 精英 | 40 | 32 | 39 |
| hand_select / card_select | 80 | 77 | 80 |
| 其他 11 种界面 | 339 | 339 | 339 |

21 个不同的决策都检查了原因：**每一个**都有一个 power 的 id，以前是 `<unknown>`，现在能识别了（`API_SUFFIXES` 修复），这正是词表更新的目的。另外 25 个决策也有 token 变化但输出完全相同：那些是新词表里**新增**的 token（例如新的 target_type），迁移时给的是旧的 `<unknown>` 行，所以输出不变，和设计一致。没有无法解释的差异。

**3. Part 4 的起点能加载。** `runs/step2-search/checkpoints/update_001052.pt` 通过 `--init-from` 的检查（151 个张量，回报缩放 var 74.6）。

## 下一步

Part 4：`--init-from runs/step2-search/checkpoints/update_001052.pt`，加上 `--search-combat --search-fights-out-of-rollout --target-kl`，重新训练宏观策略，再和 intent8h 用同一个 MCTS 打战斗来比较。
