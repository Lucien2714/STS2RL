# Step 2-p（性能）：训练进程的 HTTP 开销

*2026-10-06*

## 问题

用 10 个模拟器训练时，稳定吞吐量约 53 局/小时，与 7 个模拟器相同：lane 多了 43%，每个 lane 每步慢了 52%（1.86 → 2.82 秒）。模拟器每个约占 0.2 个核，训练进程约占 3.6 个核。瓶颈是训练进程，不是模拟器。

对运行中的训练进程用 py-spy 采样 60 秒（`runs/step2p-train`，10 个 lane）：

| 占训练进程采样的比例 | 原因 |
|---|---|
| 约 12% | `get_netrc_auth`：requests 每次请求都查找 `.netrc` |
| 约 14% | `proxy_bypass_registry`、`getproxies_registry`：每次请求都读 Windows 注册表中的代理设置 |
| 约 7% | 响应的 JSON 解码 |
| 约 20% | MCTS 每次模拟单独发送 `restore` 和 `reseed` 两个请求 |
| 各低于 2% | MCTS 的树逻辑、叶子评估、PyTorch |

## 修改

1. **`trust_env = False`（`d43fcfc`）：** 客户端不再读 `.netrc` 和代理设置。
2. **改用 urllib3（`55f36c0`、`6e112af`）：** `STS2Client` 不再使用 requests，每个客户端一个 urllib3 连接池，关闭 urllib3 自己的重试和重定向。
   - 重试规则不变：只有 GET 在传输错误时重试（最多 4 次），动作和 DELETE 从不重发。
   - 3xx 现在报错；被拒绝的动作仍带着状态报错。
   - urllib3 列为直接依赖。
3. **`restore` 和 `reseed` 合并成一个请求：**
   - 模拟器（STS2Simulator `785bf21`）：`POST /api/v1/sim/restore` 接受可选的 `reseed`（uint32，0 有效，无效值在恢复前返回 400）。恢复后 reseed，只构建一次状态，返回 `"reseeded": <seed>`。`/sim/reseed` 和 `/sim/restore` 使用同一个 `ApplyReseed`，replay 模式会记录这次 reseed。恢复成功但 reseed 被拒绝时，返回 409 并带回已恢复的状态（不回滚）。
   - 客户端：`sim_restore(point, reseed)`。响应中有 `reseeded` 且等于请求的 seed 时信任它；没有时（旧模拟器），改为单独发 `sim_reseed` 并记一次警告；值不同时报错。`CombatSearch.search` 每次模拟调用一次 `env.restore(point, seed)`，搜索结束时的恢复不带 seed。

## 验证

- **合并请求的等价性**（模拟器 subagent 测量）：合并调用与两次调用比较返回状态、隐藏状态（随机流、抽牌堆顺序）和之后 40 步。clone 模式 84/84，replay 模式 42/42；reseed 后建快照再恢复也一致（clone 32/32，replay 16/16）。有 1 个 run 的整图 digest 在第 21–22 步不同，但旧的两次调用也能复现，不是这次修改引起的。
- **测试：** 全部 792 个测试通过（修改前 756 个），ruff 无问题。新测试包括一个使用真实 socket 的本地 HTTP 服务器：40 个请求只建 1 个 TCP 连接；读超时和响应体截断时，GET 重试、POST 不重发。
- **审查：** Codex 分别审查了 3 个提交，结论都是可以合并；它提出的小改都已完成。

## 测量

`scripts/bench_client.py`，模拟器端口 15600，训练没有运行，5 轮中位数，每轮 400 个周期（restore、reseed、最多 4 步战斗），2026-10-06：

| 客户端 | 每周期请求数 | 每秒周期数 | 客户端每周期 CPU |
|---|---|---|---|
| 原客户端（requests，`trust_env` 开） | 6 | 192 | 3.05 毫秒 |
| urllib3，两次调用 | 6 | 275（+43%） | 1.06 毫秒（−65%） |
| urllib3，一次调用 | 5 | 316（+64%） | 1.02 毫秒（−67%） |

这是单个请求循环的测量，不是 10 个 lane 的训练进程。训练中的实际提速见 Step 2-p 的训练记录（`runs/step2p-fast`）。

## 部署

- 模拟器：`feat/state-snapshot` 快进到 `785bf21`，主目录重新编译（Release，0 警告）。
- STS2RL：`feat/mcts-combat` 快进到 `6e112af`。
- 原训练 `runs/step2p-train` 在 63 局时随会话重启停止（所有训练和模拟器进程都退出，日志中没有 exit 行），之后用同样的设置在 `runs/step2p-fast` 重新开始。
