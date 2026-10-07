# Step 2-q：商店里的 Foul Potion，以及被拒绝后不重复同一动作

*2026-10-06*

## 问题

`runs/step2*` 的 5040 局中有 14 局截断，**全部是同一个原因**：在商店里，agent 一直发送 Foul Potion 的 `use_potion`，游戏一直拒绝，直到停滞预算用完。

- 游戏允许在商店里对商人扔 Foul Potion：+100 金币，**不开战斗**。只有在假商人事件（fake merchant）里才开战斗。
- 游戏的检查 `FoulPotion.PassesCustomUsabilityCheck` 要求商人的物品栏是**关闭**的。
- 两个后端都通不过这个检查：
  - **mod：** 读商店状态时会自动打开物品栏，所以 `use_potion` 总是返回 "cannot be used right now"。
  - **模拟器：** 没有界面，没有 `NMerchantRoom`，所以检查总是失败。
- 被拒绝且画面没有变化时，runner 丢弃这个决策（不进 rollout），所以策略学不到任何东西，下一次又选同一个动作。13 次拒绝中有 12 次是连续重复。

## 修改

### 模拟器（STS2Simulator `d985824`）

Harmony prefix 补丁 `FoulPotionAtMerchant`，作用在 `FoulPotion.get_PassesCustomUsabilityCheck`：

- 在战斗外、`MerchantRoom` 里，返回 true。
- 战斗中和假商人事件中，仍然使用游戏自己的检查。
- `OnUse` 不变：它跳过不存在的界面节点，然后加金币。

### mod（STS2MCP `09c4938`、`fa284de`，分支 `feat/shop-foul-potion`）

- `use_potion` 在战斗外使用 Foul Potion 时，先关闭商人的物品栏（`CloseMerchantInventoryForFoulPotion`，点击物品栏的返回按钮，和 `proceed` 的做法相同），再做游戏自己的检查。
- 返回按钮不可用时（输入被阻止，或上面有别的界面），返回错误，不强制关闭。
- 不主动重新打开物品栏：下一次读状态时会自动打开，所以 `inventory_open` 和 `can_proceed` 不变。
- 文档：`raw-full.md`、`raw-simplified.md`。审查后修正：假商人战斗之后不回到事件，而是奖励界面，然后地图。
- **状态：已编译（0 警告、0 错误），没有部署，没有推送，还没有在真实游戏中测试。** 手动测试清单：`docs/testing/foul-potion-shop.md`（在 mod 仓库中）。

### STS2RL（`f389cbe`、`f5b1474`、`a10909b`）

1. **`NON_COMBAT_POTIONS`：** 先临时去掉 `(FOUL_POTION, shop)`（`f389cbe`）；两个后端都支持之后，恢复（`a10909b`）。所以候选集合和修改之前相同，已经清洗的 BC 数据集仍然有效。
2. **runner 排除重复被拒绝的动作（`f389cbe`，审查后改为 `f5b1474`）：**
   - 同一个动作在**没有变化的画面**上**连续被拒绝 2 次**，就把它排除，直到画面变化或有动作被接受。
   - 第 1 次拒绝仍然重试，因为它常常只是时序竞争。只拒绝 1 次就排除，会把一次竞争变成另一个选择，常常不可撤回（例如 `play_card` 被排除后只剩 `end_turn`）。
   - 所有候选都被排除时，按原来的方式选择，停滞预算仍然会截断这一局。
   - `LegalActionProvider` 不变，候选集合仍然只由状态决定。排除通过 `Agent.choose_action(..., exclude=...)` 传入。
   - PPO 在缩小后的集合上记录决策，所以 log probability 对应真正提供的候选。搜索 agent 有排除时不记录这个决策。
3. **搜索根节点（`f5b1474`）：** 先按完整参数去掉被排除的动作，再按 key 分组。之前顺序相反：手里有两张同名牌时，被拒绝的那张会一直是分组的代表，访问次数和子树都来自它。

## 验证

- 全部 813 个测试通过，ruff 无问题。
- Codex 审查了 STS2RL 的提交和 mod 的提交；提出的问题（第 1 次拒绝就排除、根节点分组顺序、mod 文档中假商人战斗之后的流程）都已修改。
- 模拟器主目录已快进到 `d985824` 并重新编译（Release，0 错误）。下一次训练在模拟器上可以使用这个药水。

## 待做

- 用户按 `docs/testing/foul-potion-shop.md` 在真实游戏中测试 mod，之后再部署。
- mod 部署之前，真实游戏仍然会拒绝这个药水；runner 现在会在 2 次拒绝之后排除它，不会截断这一局。
