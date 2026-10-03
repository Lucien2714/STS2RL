# 词表更新：以游戏 v0.107.1 本身为准

*2026-10-03*

这一步不属于 MCTS 计划本身，是按要求插入的：用反编译游戏客户端的方式更新词表。它会改变词表指纹，所以放在 Step 1 之前做，之后训练的 checkpoint 都基于新词表。

## 做了什么

- **从游戏本身导出 id**（STS2Simulator `Sts2Sim.Spike --export-vocabulary FILE`）：模拟器加载的就是 Steam 上同一个 v0.107.1 构建（`59260271`）。导出内容：
  - `ModelDb` 里每一类内容的 id 和英文名：卡牌 578、遗物 297、能力 273、药水 65、怪物 102、遭遇 80、事件 65（含远古事件，并从本地化键 `EVENT.pages.PAGE.options.OPTION.title` 读出页面和选项）、附魔、苦难、宝珠、角色、幕、修饰器；
  - 意图（`IntentType` 枚举）和营火选项（每个 `RestSiteOption` 子类的 `OptionId` 和标题）；
  - 代码里固定表的取值来源：`MapPointType`、`TargetType`、`CardType`、`CardRarity`/`RelicRarity`/`PotionRarity`、`PowerType` 枚举，以及 `NCardGridSelectionScreen` 和 `CrystalSphereItem` 的所有子类名。
- **反编译 STS2MCP mod**（`mods/STS2_MCP/STS2_MCP.dll`），确认它怎么写这些字段：`target_type`、`rarity` 是枚举的 `ToString()`；`screen_type` 对 4 种界面用固定字符串（transform/upgrade/select/simple_select），其余直接发类名；`hand_select` 发 `mode`（simple_select/upgrade_select）；水晶球物品发类名；敌人只有 `entity_id` 和 `name`，**没有 `id`**。
- **合并脚本** `scripts/update_vocabulary.py`：只添加现在解析不了的 id，不删除任何 id，并报告代码里固定表缺少的值。可以在每次从 spire-codex 刷新表、或者游戏更新之后重新运行。

## 发现的缺口和修法

| 表 | 问题 | 修法 |
|---|---|---|
| powers | 游戏发 `FLEX_POTION_POWER`，表里是 `FLEX_POTION`；按 id 永远对不上，只能靠显示名回退。spire-codex 把 5 个能力都叫 "Temporary Strength"、8 个叫 "Temporary Strength Down"，这些名字有歧义被丢弃，于是 **15 个真实存在的能力被读成 `<unknown>`**（Flex Potion、Speed Potion、Dark Shackles、Setup Strike 等） | `API_SUFFIXES`：只在 id 和别名都对不上时，去掉 `_POWER` 后缀再查一次。15 个能力全部落到表里已有的那一行，**不需要新增行** |
| powers | 游戏里有两个都显示为 "Monarch's Gaze" 的能力，旧的按名字解析把减力量的那个并进了主能力那一行 | 后缀规则按 id 区分开了，这是一处纠正 |
| monsters | 敌人没有 `id`，只能靠名字。Test Subject 的名字带一个每场不同的编号（"Test Subject #C55"），Tough Egg 显示为 "Hatchling"，都解析不了 | `_monster_keys`：id、名字之后再试 `entity_id` 去掉序号的部分（`TEST_SUBJECT_0` → `TEST_SUBJECT`），放在最后，按名字能解析的不受影响 |
| rest_options | 缺 CLONE、COOK、HATCH、KINDLE、MEND | 加进固定表；HEAL 本来就通过标题 "Rest" 落到 `rest`，不另加一行 |
| map_node_types | 缺 Ancient（人类录像里出现 799 次）、Unassigned | 加进固定表 |
| target_types | 缺 TargetedNoCreature（235 次）、Osty | 加进固定表 |
| selection_types | 缺 NCombatPileCardSelectScreen（828 次）、NDeckEnchantSelectScreen（208 次）、upgrade_select（115 次） | 加进固定表；mod 会改名的 4 个界面类名永远不会发出来，不加 |
| crystal_item_types | 只有 Gold，缺 CardReward、Curse、Potion、Relic | 加进固定表 |
| card_types、power_types | 缺 None | 加进固定表 |
| intents、modifiers | 缺 HIDDEN、CHARACTER_CARDS | 加进 JSON 表 |

只在游戏测试代码里存在的 `MOCK_*`、`DEPRECATED_*` 不会出现在实际游戏中，跳过。表里有、但这个构建已经没有的 id（13 个怪物、`GLASS_ORB`、`BINARY` 等）保留，多一行没用的不影响任何东西，删掉一个 API 仍会发的 id 却会变成静默的 `<unknown>`。

## 测试

- 对游戏导出的每一个 id（排除测试用的），比较修改前后解析到的行：**新增可解析 22 个，原来能解析的 0 个丢失，只有 1 个改变**（就是上面 Monarch's Gaze 的纠正）。
- 合并脚本复查：没有任何 JSON 表的 id 解析不了；固定表也没有缺失。
- **17 局人类录像（v0.107.1，13,285 步）全部 tokenize，词表未命中 0 个**（修改前 16 种）。另有 3 局 10-02 新录的是旧的 schema 1 格式，解析器不支持，没有检查。
- 模拟器契约回归 `sim_harness.py --contract-episodes 200`：0 个动作错误、0 个故障，**parity 第一次 PASS**（模拟器独有的词表未命中从 3 个降到 0）。
- STS2RL：620 个测试通过（新增 6 个：能力后缀、Monarch's Gaze 区分、后缀只是回退、新版营火选项和意图、后缀进入指纹、战斗单位按 entity id 解析；修改 1 个：原来拿 `STRENGTH_POWER` 当"解析不了的 id"的例子已经不成立），ruff 通过。

## 影响

- 词表指纹改变（`FINGERPRINT_VERSION` 3 → 4），**所有现有 checkpoint（包括 intent8h 和 `bc_best.pt`）都不能再加载**，需要重新训练。BC 的清洗数据只依赖动作层，不需要重新清洗。
- CLAUDE.md 里"The API does not spell ids the way the bundled tables do"一节补充了后缀规则、战斗单位的解析方式和更新流程（CLAUDE.md 在工作区根目录，不在任何 git 仓库里）。

## 闸门

通过。
