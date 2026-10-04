# Step 2-a/b/c：分层 agent（MCTS 打战斗，PPO 学其余决策）

*2026-10-03*

## 做了什么

- **2-a，`CandidatePPOAgent.choose_external(state, action, lane)`**（`agents/ppo.py`）：动作由外部（搜索）选出，agent 只记录它，照常计算 critic 的 value。`_PendingDecision` / `_RolloutStep` 新增 `policy_trainable`；`update()` 的 `policy_mask` 和 `_optimize_minibatch` 改为调用 `_policy_step(step)`：候选数 > 1 **且** 是这个策略自己选的。所以外部动作和强制步的处理完全一样：参与 GAE 和 critic loss，不参与 policy loss 和 advantage 归一化。理由：战斗步必须留在轨迹里，否则战斗之后的奖励传不回战斗之前的地图选择；但这个动作不是从当前策略采样的，没有可以形成 ratio 的 log 概率。`LaneView` 同样转发。
  - `GameAction` 没有 `__eq__`，所以外部动作按 `to_dict()` 在候选里查找，返回候选里的那个对象。
- **2-b，`SearchCombatAgent`**（`search/agent.py`）：包装一个 PPO lane 和这个 lane 自己的 `GameEnv`。
  - 默认**所有战斗**（普通、精英、boss）都由搜索打，`--search-rooms` 可以缩小范围；其余状态交给 PPO。
  - 战斗中的手牌选择、牌堆选择这类界面会阻塞游戏线程、不能做快照，用上一次搜索的子树回答（和 `play_fight` 一样），也作为外部动作交给 PPO。战斗的房间类型只在战斗主界面上有，所以记住"当前在一场要搜索的战斗里"，直到战斗结束。
  - 动作被拒绝且界面没动时（`discard_decision`），子树指针退回到选这个动作之前的位置，重试时从同一个节点回答。
- **2-c，`SearchDecisionRecorder`**：每个搜索过的决策写一行到 `<run-dir>/search/decisions.jsonl.gz`，格式就是 `offline.clean.Decision.to_json()` 的格式（`expert_index` 是搜索打出的动作），再加 `target_distribution`（根节点的访问比例，和候选一一对应）和 `search_values`。每行一个 gzip member，多个 lane 只需要一把锁就能追加，进程被杀最多丢正在写的那一行。Step 3 只需要让读取端认识 `target_distribution`。
- **训练入口**：`sts2rl-train` 新增 `--search-combat`、`--search-rooms`、`--search-simulations`（默认 50）、`--search-depth`（默认 2）、`--search-weights`（默认 Step 1-b 的权重）。这些设置写进 `TrainingConfig`，保存在 checkpoint 里，resume 时不能改（和 `--backend` 一样：战斗被搜索打过的 run 和没有的是两个不同的实验）。`--search-combat` 必须配 `--backend sim`。旧的 plan 没有这些字段，加载时取默认值（关闭）。

## 测试

- `tests/test_ppo_agent.py`：外部动作只记给 critic（`policy_trainable=False`，下标正确）；全是外部动作的 rollout 更新后 `policy_loss` 和 `entropy` 都是 0、`value_loss` > 0；不在候选里的动作被拒绝；`LaneView` 把外部动作交给自己的 lane。
- `tests/test_search_agent.py`（新）：默认搜索所有战斗；不搜索的战斗和战斗以外 PPO 自己选；精英战被搜索并作为外部动作交给 PPO；战斗中的选牌界面用子树回答、不再搜索；被拒绝后子树回退；离开战斗后交回 PPO；记录的行候选、`expert_index`、`target_distribution` 都正确；学习相关的调用原样转发。
- `tests/test_training_config.py` / `tests/test_training_cli.py`：搜索需要模拟器；房间名校验；设置能序列化往返；旧 plan 能加载；CLI 选项进入 plan；默认关闭。

全量回归 656 个通过，ruff 通过。

## 冒烟测试（2-d 的第一步）

从随机权重开始，7 个模拟器，7 局，MCTS 打所有战斗（`runs/step2-smoke`）：

| 局 | 层数 | 步数 | 秒 |
|---|---|---|---|
| 1 | 6 | 80 | 100 |
| 2 | 12 | 118 | 121 |
| 3 | 14 | 183 | 198 |
| 4 | 17 | 202 | 231 |
| 5 | 17 | 237 | 235 |
| 6 | **25** | 283 | 239 |
| 7 | **24** | 280 | 249 |

- 没有报错、没有截断、没有 action error；5 次 PPO 更新，value_loss 0.07–0.33，梯度范数 0.7–3.7。
- 搜索了 742 个决策（普通战斗 567、精英 90、boss 85），全部写进了记录文件。
- **actor 是随机权重，地图、选牌、营火全是随机选的，7 局里仍有 2 局打过了第 1 幕 boss。** intent8h 训练 8 小时后平均只到 13 层左右。这再次说明战斗操作的影响有多大。
- 速度：一局到 25 层约 4 分钟，7 个模拟器并行，相当于每小时约 100 局。打得越远，一局越长，后面会更慢。

## 闸门

通过。下一步：Step 2-d 的正式训练。
