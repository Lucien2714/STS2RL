# Step 2-o：宝箱必须先拿遗物，再离开

*2026-10-06*

## 问题

Step 2-n 的 600 局训练中，策略学会了在宝箱房不拿遗物就离开。

- `LegalActionProvider._treasure_actions` 同时提供 `claim_treasure_relic` 和 `proceed`。
- 拿遗物再离开是 2 步，直接离开是 1 步。reward 不直接奖励遗物，每步扣 0.01，所以直接离开总是便宜 0.01。遗物的价值只在之后的楼层奖励中体现，延迟长，噪声大。
- 在 53 个人类宝箱界面上，P(拿遗物) 在第 0–72 次更新是 1.00，第 132–180 次更新降到 0.02–0.17。
- 第 1 幕 boss 前的遗物数从 4.1 降到 3.2，第 1 幕 boss 通过率从 68% 降到 60%。

这和以前奖励界面上 27/28 次选 `proceed` 是同一个陷阱（`CLAUDE.md`："A reward screen is claimed out, not walked away from"）。

## 修复前的检查

`CLAUDE.md` 要求先确认：拿走遗物后，宝箱是否清空。如果宝箱不清空，不提供 `proceed` 会卡住这一局。

- **人类记录**（真实游戏客户端，20 局）：65 次宝箱访问。61 次拿遗物，61 次之后的状态都是 `treasure`，`relics` 为 `[]`（48 次）或字段消失（13 次），`can_proceed` 为 true，接着 `proceed` 进入地图。1 次直接跳过（Lantern，第 26 层）。3 次宝箱一开始就是空的（遗物 Silver Crucible）。
- **模拟器代码**：`StateBuilder.cs` 在拿取后返回空列表；`ActionHandler.cs` 在未拿取时 `proceed` 跳过遗物。
- **模拟器实测**：2 个 seed，拿取后 `relics` 为 `[]`，之后只剩 `proceed`。

结论：单人模式下，拿遗物会清空宝箱。只提供拿取不会卡住。

## 修复

`src/sts2rl/agents/action_space.py` 的 `_treasure_actions`，规则与 `_reward_actions` 相同：

| 宝箱状态 | 提供的动作 |
|---|---|
| `relics` 是非空列表 | 只提供能读出 index 的 `claim_treasure_relic` |
| 非空列表，但没有能读出的 index | 不提供动作，runner 重新读取状态 |
| `relics` 存在但不是列表 | 不提供动作 |
| `relics` 为 `[]` 或不存在，且 `can_proceed is True` | 只提供 `proceed` |
| 其他（包括 "Opening chest..."） | 不提供动作 |

只有一个遗物时，拿取是单候选（强制）步，之后的 `proceed` 也是。宝箱界面因此不再训练 actor。没有修改 `GameEnv`，没有加 reward 项。多人模式的竞价字段没有验证，规则不覆盖它。

## 审查

- Codex 审查了设计和 diff。第一次审查的结论是“小改后提交”：
  - `relics` 存在但不是列表时，原代码会提供 `proceed`，现在改为不提供动作；
  - docstring 的两处说明修正，并注明单人范围；
  - 补充类型边界的测试。
- 修改后全部按清单完成。

## 测试

- `tests/test_action_space.py`：原有的宝箱用例改为只期望拿取；新增参数化测试 `test_a_treasure_chest_is_claimed_out_before_proceed`，17 个用例（单个、多个遗物，空列表，字段缺失，读不出 index，混合列表，`relics` 为 dict/字符串/`None`，`can_proceed` 为非布尔值，有遗物但不能离开）。"Opening chest..." 的旧测试不变。
- 全部 735 个测试通过（修改前 718 个），ruff 无问题。

## BC 数据

- `data/bc/v6` 的 53 个宝箱决策不再与动作空间一致，`BCDataset` 拒绝 `v6`（`StaleDatasetError`）。
- 新数据集 `data/bc/v7`：保留 10771 个决策（v6 为 10824）；强制步 2134（+52）；不匹配 250（+1，即人类那一次跳过）。`--verify` 通过。之后的 BC 训练和 `sts2rl-surgery outputs` 应使用 `v7`。
- `gameplay_records` 中 3 条 2026-10-02 的记录是 `schema_version` 1，清洗程序不读它们。按用户要求已删除，现在剩 17 条，与 `v6`、`v7` 使用的记录相同。

## 下一步

在新分支 `feat/pbrs` 上测试基于势函数的 reward shaping（PBRS），解决营火总是休息的问题（见 Step 2-n 的结论）。A/B 的两组都带本修复。
