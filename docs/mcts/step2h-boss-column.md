# Step 2-h：当前幕 boss 特征（`act_boss`）和 boss run（`runs/step2h-boss`）

*2026-10-05*

## 做了什么

模型以前只在地图界面上能看到这一幕的 boss（`map.boss`），而选牌、营火、商店这些决定牌组的界面都看不到。按指示改了 API，再加一个特征：

- **STS2MCP mod**（`E:\personal\dev\STS2MCP`，分支 `feat/run-boss`，提交 `0318851`）：`BuildRunInfo` / `AddRunBoss` 在每个界面共用的 `run` 块里加上 `run.boss`（有两个 boss 的幕再加 `run.second_boss`），id 和名字与 `map.boss` 相同。对 `D:\sts2depot\sts2_1` 编译通过，0 个警告。**还没有在真实游戏里测过**（需要登录 Steam）。
- **STS2Simulator**（分支 `feat/state-snapshot`，提交 `38fc195`）：`StateBuilder.RunInfo` / `AddRunBoss`，同样的字段。
- **STS2RL**（`7765a19`）：`encoder/schema.py` 的 `GLOBAL_CATEGORICAL` 加 `("act_boss", "encounters")`；tokenizer 用 `lookup_first("encounters", (id, name))` 读 `run.boss`。没有这个字段（这一幕还没有 boss，或旧的录像）时是 PAD，表示"不知道"，不猜。`docs/raw-full.md` 补了 `run.boss`，并把 `map.boss` 的例子改成 `VANTOM_BOSS`。

模型迁移用 Step 2-f 的 surgery：`step2e-stable` 在第 4 个 seed 周期结束（600 局，`update_000172`）时按指示停止，`sts2rl-surgery migrate` 把它迁移到新 schema（新列的嵌入表从 0 开始，输出不变，`outputs`/`compare` 检查通过；第一次用错了旧 spec `715b65b`，形状检查拒绝了，换成正确的 `4e1a30f`），然后 `--init-from` 开始新的 run。

迁移时发现 `resolve_source` 对相对路径的 run 目录会多拼一层 `checkpoints/`，换 run 前的输出检查正确地拒绝了启动；修复并加测试（`bab2de1`）。

## 结果（656 局，193 次更新，10-05 03:02 ~ 09:11）

| 周期 | 局 | 平均 floor | 到第 17 层 | 最高 |
|---|---|---|---|---|
| 1 | 1–150 | 19.7 | 110 | 48 |
| 2 | 151–300 | **20.6** | 113 | 42 |
| 3 | 301–450 | 20.0 | 110 | 48 |
| 4 | 451–600 | 18.8 | 97 | 48 |
| 5（部分） | 601–656 | 19.5 | 39 | 40 |

- 一开始就在 `step2e-stable` 的水平（第 4 周期 19.6），没有因为换 run 掉下去；第 2 周期最高，之后回到 19–20，**平台期约 19.7**。
- boss 特征对选牌的影响很小：在人类录像的界面上，同一个界面换不同的 boss，选牌概率最多变 10% 左右。模型看到了 boss，但还没学会按 boss 构筑牌组。

## 下一步

在 boss 战里多花搜索预算（Step 2-i）。
