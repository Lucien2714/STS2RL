# 开发日志

2026-10-06 21:00 (PT) 之前的条目根据 git 历史、docs/ 下的文档和 runs/ 下的驱动日志重建。

### 2026-06-16 22:16 (PT) · main
- 做了什么：初始化项目（6086a52，2026-06-09）。合并 PR #1（c0254bb，来自 feat/hand_deck_changes）：修复战斗中选牌卡住（564a7a3）；新增 Card 类存卡牌（90d0c2c）；统一 torch 导入（62c58e7）；健壮性修复（fa1cabc）。
- 结果/数据：无实验数据。测试数量未记录。
- 遇到的问题：战斗中的卡牌选择和手牌选择会卡住。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-06-20 02:51 (PT) · behavioral-cloning-pretraining
- 做了什么：40949a1：候选动作 schema 从 v1 升到 v2，加入 select_card / confirm_selection / cancel_selection；新增 BattlePPOAgent；加入 TensorBoard 记录。0c0632e：行为克隆预训练（sts2rl-pretrain），按 action_key 把人类动作匹配到一个合法候选，用交叉熵训练。2562aa7：写 docs/pretraining.md。
- 结果/数据：无实验数据。测试数量未记录。
- 遇到的问题：从零训练的 DQN 塌缩成"总是结束回合"。
- 下一步：先做 BC，再用 RL 微调。
- 需要 Lucien 决定的事：无

### 2026-06-22 01:52 (PT) · fix/battle-agent-highsev
- 做了什么：7983a94 修复三个高严重度错误：DQN 平局时总选第一个候选（end_turn）；多客户端 PPO 共用一个 pending 槽，互相覆盖；GAE 在交错的轨迹上计算。改为每个客户端一个 rollout collector，GAE 按轨迹分别计算。b40d309：抽出 BattleStateEncoder，DQN 和 PPO 共用 CandidateActionAgent；删除无用的 config/、algorithms/；新增 discard_potion 动作；schema 升到 v3。
- 结果/数据：测试通过（49 个；b40d309 之后 50 个）。维度 4834/102/4936，旧 checkpoint 被拒绝。无实验数据。
- 遇到的问题："总是结束回合"的可能原因：候选里 end_turn 排第一，Q 值相等时 max 返回第一个。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-06-23 00:25 (PT) · main-deprecated
- 做了什么：把候选动作框架推广到非战斗界面（地图、奖励、商店、营火、事件）（9b6385c、d935626、4487c94）。CandidateEncoder 改为 ABC（ae658d8）。加入 ruff 并统一格式（6c66244）。新增 CardModelEncoder（16 维卡牌嵌入）和保存/加载（ebc62b1、b1fc6a5）。写 ADR 0001–0005 和卡牌嵌入设计说明（b5662e1）。战斗奖励和整局奖励分开（09b3bd7）。每个界面的 agent 接入预训练和评估（6402441）。
- 结果/数据：测试通过（64 个；加入卡牌嵌入后 72 个）。无实验数据。
- 遇到的问题：代码缩进不统一（2 格和 4 格混用）。CandidateEncoder 原来是非正式接口，缺方法时只在调用时报错。
- 下一步：CardModelEncoder 还没有接入编码器和网络（ebc62b1）。
- 需要 Lucien 决定的事：无

### 2026-08-30 18:17 (PT) · main-deprecated
- 做了什么：agent 架构重设计阶段 0–2：合法动作枚举移到 action_spaces/（7b82de4）；策略网络移到 models/policies.py（a299b3f）；每个界面的 agent 子类换成注册表（b217852，ADR-0007）。sts2rl-train 新增 --screen（dd130c3，在 fix/battle-agent-highsev 上）。加入学到的卡牌策略，训练和评估共用步进逻辑（2331237）。
- 结果/数据：无实验数据。重构前后候选顺序、特征向量和 checkpoint 一致（golden capture 和固定种子的 checkpoint 加载验证）。测试数量未记录。
- 遇到的问题：先导入 sts2rl.encoders 再导入 agents 时有循环导入，会崩溃（7b82de4 修复）。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-08-30 23:14 (PT) · main-refactor
- 做了什么：在新的环境契约上重建项目。f3d9d7a：环境边界使用类型化的 GameAction。83dac65：加固 STS2MCP HTTP 客户端。db54dac：删除旧的 agent、算法、流程、训练和评估层。ba56443：定义 GameEnv 的原始契约（RawState、EnvStep、ResetSpec）。6969c05：菜单导航抽成 ResetController。0e32722：可变动作 PPO agent 的基础。
- 结果/数据：无实验数据。测试数量未记录。
- 遇到的问题：未记录。
- 下一步：在新契约上重建学习层（db54dac）。
- 需要 Lucien 决定的事：无

### 2026-09-02 11:32 (PT) · main-refactor
- 做了什么：结构化编码器。e6fe5c0：特征编码契约移到 encoder 包。0e21985：确定性词表。53051b9、a2c7102、cd89b4e、df5920e：定义 token，tokenize 实体，动作链接到实体，tokenize 完整地图 DAG。1af4fc4：实体 transformer。2b19dff：地图 DAG 编码器。677da44、8be5835：组合状态和动作编码，agent 拿到完整观测。6be1b3f、019cc88：用候选 PPO 训练结构化编码器。
- 结果/数据：无实验数据。测试数量未记录。
- 遇到的问题：未记录。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-04 22:39 (PT) · main-refactor
- 做了什么：可恢复的训练运行时（ec05bc4）。架构清理（8c087ff、4b12651）：删除 DQN 时代的模块；单候选状态跳过 rollout，奖励并到前一个决策；没有合法动作时截断，不再抛错；加入 PPO minibatch。b80b2e2：候选编码和地图编码改为批处理；计数器只存一份。766381d：奖励缩放到同一数量级，补测试。89cf193：tokenizer 和编码器共用 encoder/schema.py。1d561d9：文档和代码对齐，补上 sts2rl-train 入口。
- 结果/数据：地图编码逐层批处理和逐节点循环的最大误差 4.8e-07（105 节点地图）。15x7 地图上前向 14.0 → 6.5 ms，反向 54.5 → 15.9 ms。checkpoint 格式升到 2。测试数量未记录。
- 遇到的问题：胜利奖励 200，每步项约 1，value loss 主导共享主干。计数器存了三份，五个方法只为检查它们一致。字段名和词表在两个文件里按位置配对，加列会被静默截断。文档里的 sts2rl-train 命令没有入口，无法运行。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-05 00:20 (PT) · main-refactor
- 做了什么：865a524：对着真实游戏客户端修复 reset 的五个问题（角色选择读错位置；按"开始"后当作瞬时完成；返回过渡状态 unknown；game_over 离不开；/player-detail 不存在时每局都失败）。d99646f：代码按官方 API 参考 docs/raw-full.md 对齐；动作响应自带状态，step 不再额外 GET；ResetSpec 新增 ascension。
- 结果/数据：实测两局，各 60 步，无 action error，第一局死亡，第二局开新局。20 步的一局：24 个 POST、4 个 GET。等级 10 的存档可以从等级 0 开局。测试数量未记录。
- 遇到的问题：代码原来依据的 API 文档描述的是另一个 mod。卡牌字段减少，张量宽度和词表指纹改变，旧 checkpoint 不能加载。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-05 23:12 (PT) · main-refactor
- 做了什么：513cb91：读取新的 GET /api/v1/player，加入整局牌组（deck 区）。1210e53：实时审计后修复：状态 id 按显示名回退（lookup_first）；意图的 CamelCase 回退；买不起东西的商店允许 proceed；读取 battle.round。5397c80：跟进 API 的第二次更新（统一 Card Object、current_upgrade_level）；新增 --ascension；记录被拒动作的消息。3bfccf9：从 spire-codex 刷新游戏数据表，新增 affliction_id 列。
- 结果/数据：GET /player 33 ms；22 步一局中 14 次读取共 466 ms（18.2 秒的 2.6%）；25 张牌组使编码器前向 1.17 → 1.63 ms。刷新表后实测 170 个状态：0 个未解析值、0 个 action error、0 个 tokenize 失败；两局训练到第 17 层，无被拒动作。测试数量未记录。
- 遇到的问题：战斗中的 buff/debuff 几乎全读成 <unknown>（STRENGTH_POWER 对 STRENGTH）。多词意图（DebuffStrong）解析不了。买不起东西的商店是死路，150 步审计全卡在那里。一局中 2 次被拒动作在之后 390 步里无法复现。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-06 03:12 (PT) · main-refactor
- 做了什么：c85276d：rollout 跨局累积，满了才更新；选择界面不再提供已选的牌，满了不再提供选牌；去掉商店 proceed 的旧变通。f2930fb：BattleProgressReward 换成 RunProgressReward：每进一个节点 +1，每个 boss +10，每步 −0.01（当时还有 HP 项 0.03）。
- 结果/数据：33 局实测：rollout_size 256 在 33 次更新中只达到 1 次，中位数 65，最小 10；梯度范数 180–520（max_grad_norm 0.5）。旧奖励下回报到 +131，value_loss 69–1580，policy_loss ±0.5。新奖励下一幕约 26，全胜约 80。测试数量未记录。
- 遇到的问题：每局都更新一次，advantage 只在约 10 个样本上归一化。一局在选择界面耗了 400 步，奖励为 0。5 次 "Rest site room is not open" 拒绝需要实时样本才能区分。
- 下一步：value loss 的规模问题由新奖励处理（f2930fb）。
- 需要 Lucien 决定的事：无

### 2026-09-06 03:47 (PT) · main-refactor
- 做了什么：cfdc420：每个被接受的动作之后暂停 action_delay_seconds（默认 0.1 秒），--action-delay 可调。a36c01d：训练用固定 seed 池循环，另设 holdout 池；custom_run 的修饰器按 ResetSpec.modifiers 对账；reused_run 记录接管的 run。28637b9：分析脚本，按块看趋势、按 seed 比较前后半段。
- 结果/数据：暂停不降低吞吐：有暂停 0.40 s/步，无暂停 0.49 s/步（n=3）。seed 7NKRVDBV 多次重启 Neow 界面相同；三个训练进程中得到相同的 50 步、第 4 层、+0.100 轨迹。测试数量未记录。
- 遇到的问题：每局随机开局，回报里混入运气。custom_run 会记住修饰器勾选，跑错修饰器也不会报错。
- 下一步：用 holdout seed 区分"学会爬塔"和"记住地图"（a36c01d）。
- 需要 Lucien 决定的事：无

### 2026-09-06 15:14 (PT) · main-refactor
- 做了什么：10506fe：多个游戏客户端同时训练（--ports），每个客户端一条 lane，GAE 按 lane 计算；前向、rollout、优化器用一把锁。c82cbb8：奖励去掉 HP 项，只剩节点、boss 和每步扣分。
- 结果/数据：单客户端 3 局 seed 实验复现之前的数字（92 步 rollout，value_loss 0.546）。多客户端只由测试覆盖。测试数量未记录。
- 遇到的问题：两局交错进一个 rollout 时，GAE 会跨游戏 bootstrap，且不报错。HP 计分让 agent 每个营火都休息，不升级（奖励被钻空子）。
- 下一步：多客户端实测需要第二个游戏安装（mod 端口在 STS2_MCP.conf 里）。
- 需要 Lucien 决定的事：无

### 2026-09-06 21:38 (PT) · main-refactor
- 做了什么：05c6817：GET 在连接断开时重试，动作不重试；resume 失败时报出可能原因。bcb18a6：只在药水栏满时提供 discard_potion。d5d3579、7bd3913、7b88300：菜单竞态：重复的画面先等待再判断卡住；被拒的点击按状态判断是否已经前进；整个 reset 失败时重试（max_reset_attempts，默认 4）。
- 结果/数据：两个长 run 在 GET 上断开（WinError 10061、10054），相隔几十局。一个 373 局的 run 死于 "Reset menu stopped making progress"。菜单竞态先后在第 374、406、501 局终止长 run。MAX_TRANSITIONS 10 → 16。测试数量未记录。
- 遇到的问题：4 个客户端在同一台机器上会掉 TCP 连接。弃药水无条件提供，agent 会扔掉药水。存档存在时主菜单换成 continue/abandon_run。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-07 00:43 (PT) · main-refactor
- 做了什么：44cd1df：等待战斗进入出牌阶段的预算改为 12 次刷新，退避上限 1 秒（约 10 秒）。a4e3a5d、b27a2f1：一局中 STS2ClientError 只丢掉这局和这条 lane 的轨迹，连续失败 3 次的 worker 退出，其他错误仍然直接失败。632c109：动作后暂停改为 0.2 秒。
- 结果/数据：601 局中 103 局截断，某个 50 局块达到 60%；reused_run 100 次对应 103 次截断，都从第 403 局开始。测试数量未记录。
- 遇到的问题：原来的等待预算只有 1.75 秒；策略到第 9–14 层后，多敌人的回合更长。三次客户端崩溃各自终止了跑了几小时的 run。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-07 04:05 (PT) · main-refactor
- 做了什么：e7254e1、c2bbfe0、75105bf：奖励界面必须领完。金币、遗物、药水先单独提供（强制步）；之后才提供卡牌奖励；选卡或跳过后，GameEnv 自动发送 proceed。
- 结果/数据：追踪的策略在 28 个奖励界面中 27 次选 proceed，925 个决策中 0 次进入 card_reward。实测：金币 99 → 119，药水进栏，proceed 进入地图。测试数量未记录。
- 遇到的问题：第一版不提供 proceed 会造成死循环：跳过卡牌后卡牌原样留在奖励界面。
- 下一步：宝箱界面有同样的问题，需要先实测拿取后宝箱是否清空（CLAUDE.md）。
- 需要 Lucien 决定的事：无

### 2026-09-07 04:54 (PT) · main-refactor
- 做了什么：75574ac：API_SPELLINGS 把 StatusCard 映射到 STATUS；新增 sts2rl-eval（训练池和 holdout 池交替评估，报告差距）。a578063：checkpoint 按优化器更新次数间隔，命名为 update_xxxxxx.pt。a8a8d44：在 on_update 回调里保存 checkpoint，这时 rollout 必然为空。
- 结果/数据：15 种意图写法中 14 种已能解析，StatusCard 是唯一未命中。三个 run 中 5–16% 的更新因为 checkpoint 强制刷新而变短（最少 13 条样本），梯度范数高 35–58%。1549 局中有 29 次 boss 胜利。测试数量未记录。
- 遇到的问题：按局数存 checkpoint 时，后期的训练量是前期的 4 倍。丢掉剩余样本会丢掉稀有的死亡和 boss 胜利。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-07 16:10 (PT) · main-refactor
- 做了什么：选择界面的循环。3771996：选卡包时一起确认。f4779c7：选牌界面填满时自动确认。d2b599e：有可选或可确认时不提供取消。6738ce9：水晶球按规则玩，只给一个动作。3a2059c：评估不计算接管的 run，打完后把 seed 放回计划。fd6e751："被拒绝且画面不变"按停滞重试，不记入 rollout；恢复 is_selected 过滤。
- 结果/数据：300 局训练中"被拒绝且画面不变"出现 123 次。评估 run 用完 10000 步上限（argmax 在循环里出不来）。测试数量未记录。
- 遇到的问题：card_select 和 hand_select 记录已选牌的方式不同；之前只查了 hand_select 就删了 is_selected 过滤，造成死循环。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-07 20:57 (PT) · main-refactor
- 做了什么：77b97ac：critic 学习缩放后的回报（_ReturnScale，滚动二阶矩）；进入 GAE 的值乘回原尺度；缩放存进 checkpoint，FORMAT_VERSION 升到 3。
- 结果/数据：相隔 8 次更新的两个 checkpoint：floor 15.0 → 8.4；value_loss 0.29 → 5.51；梯度范数 4.3 → 19.1；entropy 正常。回报从约 ±2 涨到 +57。测试数量未记录。
- 遇到的问题：策略变强后回报变大，平方损失让 critic 梯度暴涨，通过共享编码器破坏策略。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-08 04:05 (PT) · main-refactor
- 做了什么：715b65b：事件选项按 text_key（事件 + 选项 id）识别；页面单独作为 event_page 列；effects 数值按名字分桶（EVENT_EFFECT_KEYS）；will_kill_player 三态。66639d9：无渲染运行游戏客户端（start_game_clients.ps1），每个客户端独立 APPDATA，关闭后台限帧；analyze_run.py 报告吞吐和截断。4adc602：更新 docs/rewards.md。
- 结果/数据：英文事件字符串里有 92 个效果变量名。约 300 步对比：无渲染 + 关闭后台限帧 0.524 s/步，有窗口 0.608 s/步，无渲染 + 默认设置 1.166 s/步。共用存档目录时日志有 3907 次 "Cloud write failed"、28 次 "Rename failed"。私有内存：无渲染约 725 MB，有窗口 1854 MB（docs/headless.md）。测试数量未记录。
- 遇到的问题：11 个事件把所有锁定选项都叫 "Locked"，按标题会共用一行 embedding。全新存档目录不加载 mod（settings.save 不随 Steam 同步）。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-15 07:17 (PT) · main-refactor
- 做了什么：a327e2a（提交信息为空，内容来自 diff）：boss 奖励改为按"离开 boss 所在楼层"计算；缺少 run 块时不当作第 0 层；商店只提供 can_purchase 为 true 的商品；seed 池换成从游戏历史记录取的 150 个训练 seed 和 30 个 holdout seed；记录截断原因和"被接受但无效"的动作；新增 scripts/report_progress.py、start_training_run.ps1；torch 改用 cu130 源。
- 结果/数据：无实验数据（提交信息和文档没有记录这段时间的 run 结果）。测试数量未记录。
- 遇到的问题：未记录。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-19 10:51 (PT) · main-refactor
- 做了什么：aed41b2：战斗外只在观察到有效的组合上提供 use_potion（NON_COMBAT_POTIONS：Foul Potion 在商店和假商人，Blood Potion 在商店和营火）。d35105e：离线流程：sts2rl-bc-clean 把录像清洗成决策数据集，sts2rl-bc-train 克隆 actor（按 holdout 交叉熵选 epoch），sts2rl-train --init-encoder 从克隆权重开始 PPO。另在 main 上更新 README 的当前分支说明（9cedbd9）。
- 结果/数据：第一版药水规则（按 target_type 推断）下，第 118–155 局的 38 局中有 16 局因 Skill Potion 被拒循环而截断（docs/offline.md）。当时只有 1 局录像：704 步，563 个可训练决策；基线：均匀随机 17.6%，总选第一个 32.7%。不匹配从 29 降到 17，全部是有意排除的动作。测试数量未记录。
- 遇到的问题：只有 1 局录像，没有可信的 holdout，准确率只能当作管线检查。
- 下一步：录更多局（十几局以上）。
- 需要 Lucien 决定的事：无

### 2026-09-21 00:10 (PT) · main-refactor
- 做了什么：43f8c46：模拟器后端（--backend sim，STS2Simulator），一次请求开局；--sim-start-act 从幕开始的快照开局；scripts/capture_snapshots.py。e528a6e：强制步的奖励不再跨局合并，终局步不再被延长，奖励转给下一个记录的决策。
- 结果/数据：修复前 256 步的 rollout 中终局数为 0；_ReturnScale 从 1.0 涨到 33.7（真实回报 RMS 3.8）；评估 floor 8.0 → 2.5。修复后缩放稳定在约 4.3。模拟器后端：无实验数据。测试数量未记录。
- 遇到的问题：一局的第一个决策常是强制步，合并时把上一局的 done=True 改成 False，GAE 跨局 bootstrap。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-21 22:24 (PT) · main-refactor
- 做了什么：8b009eb：--sim-start-boss 从第 N 幕 boss 战的存档开局；sts2rl-eval --backend 可以用另一个后端评估。
- 结果/数据：boss1-mc5k 每局从第 1 幕 boss 战开始，3000 局胜率一直在 4.4%–6.4%，没有上升（docs/progress.md）。按快照拆开（119 个快照，每个 25 次）：98 个从未赢，17 个偶尔赢（1–8 次），4 个几乎每次赢（19–23 次）（docs/report.md）。测试数量未记录。
- 遇到的问题：恢复快照时没有重设 RNG，同一个快照每次抽牌顺序都一样，还不能排除 boss 类型和抽牌运气的影响。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-09-23 01:43 (PT) · main-refactor
- 做了什么：7a0e0fc（合并 fix-gae-boundaries，c819406）：截断的局在 finish_episode 时关闭 trace，用当时的 V 做 bootstrap；强制步记入 rollout，参与 GAE 和 critic，用 mask 排除在 policy loss、熵和 advantage 归一化之外；采样时的 value 按原始奖励单位保存。
- 结果/数据：模型和 checkpoint 格式不变。无实验数据。测试数量未记录。
- 遇到的问题：截断的局没有边界，GAE 走进下一局的奖励。多 lane 时，动作在一次更新前采样、更新后观察，value 会按新的缩放读回。
- 下一步：未记录。
- 需要 Lucien 决定的事：无

### 2026-10-02 00:57 (PT) · main-refactor
- 做了什么：4e1a30f：意图标签 "7x2" 拆成每段数值、段数和总数三列。docs/progress.md 记录训练平台期：gae-scratch、gae-bc、sync5k、boss1-mc5k、bc-v6、cmp5k 的结果和已排除的原因。
- 结果/数据：人类录像 10162 条攻击意图中 2839 条（28%）是多段。gae-scratch（从零，10,000 局）：第一块 9.5，2–3k 局 12.7–13.0，最后 12.8–13.4。gae-bc（BC 热启动，10,000 局）：10.7 / 11.7–12.4 / 12.0–13.5。sync5k（从零，12,787 局）：8.8 / 11.7–12.8 / 13.5–14.5。gae-bc 最后 750 局：250 局死在第 17 层，只有 9 局到 18 层以上，过 boss 的只有 1–5%。bc-v6（17 局录像，10824 个决策）：holdout 57%，训练 79%，随机 19%，总选第一个 26%；cmp5k-bc 第一块 10.3 对 8.6；最终评估 gae-scratch 15.1 对 11.4–12.5。holdout 和训练池接近（sync5k 14.6 对 15.3）。value_loss 0.02–0.05，entropy 0.7–0.9，return_scale 4.5 → 6.0。测试数量未记录。
- 遇到的问题：前 2–3k 局涨 3–4 层，之后 7–10k 局只多涨 0.5–1 层。第 1 幕 boss（第 17 层）是墙。BC 过拟合，只在前期有效。意图列变宽，所有旧 checkpoint（包括 bc_best.pt）不能加载。
- 下一步：修复后从零训练一个 run，和 gae-scratch 对比；评估要对多个 checkpoint 取平均。
- 需要 Lucien 决定的事：下一步方向：分层方案（MCTS 打战斗、PPO 学宏观决策）或 RUDDER 类信用分配算法，progress.md 把两者都列为候选。

### 2026-10-02 09:03 (PT) · main-refactor
- 做了什么：用 4e1a30f 的代码从零训练 8 小时（runs/intent8h），每个阶段取最后 3 个 checkpoint，在训练池和 holdout 池上评估，和修复前的 gae-scratch 对比。写阶段报告 docs/report.md（未提交）。这个阶段没有提交，时间取自 intent8h 的 metrics 文件最后写入时间。
- 结果/数据：5 个阶段平均：修复后 13.15（训练池）/ 13.52（holdout），修复前 12.99 / 13.17，没有一致的差别。同一阶段相邻 3 个 checkpoint 的评估最多差 4.8 层（10.3 / 13.7 / 15.1）。
- 遇到的问题：多段意图修复没有改变 boss 瓶颈。评估噪声大，单个 checkpoint 的评估不能下结论。
- 下一步：report.md 第 9 节的分层方案：Step 0 模拟器 fork 和重设 RNG；Step 1 在 boss 快照上比较搜索和 actor；Step 2 MCTS 打战斗、PPO 学宏观决策；Step 3 蒸馏。
- 需要 Lucien 决定的事：无

### 2026-10-02 23:17 (PT) · feat/mcts-combat
- 做了什么：Step 0-pre（48ca0e6）：模拟器（STS2Simulator 0c9040b）用"重放请求日志"实现分支点（/sim/snapshot、/sim/restore、DELETE /sim/snapshot/{id}）。STS2Client 新增 sim_snapshot / sim_restore / sim_release。新增端到端检验脚本 branch_check.py。
- 结果/数据：20 个快照全部通过（恢复后状态一致，重放路线每步哈希一致，绕路后无残留）。恢复约 1.8 ms + 每条日志 0.12 ms，一场 boss 战约 6–16 ms。Python 端每步中位数 1.3–2.5 ms。测试通过（613 个）。契约回归 200 局：0 个动作错误、0 个模拟器故障。
- 遇到的问题：随机策略在 boss 战死得快，检验路线偏短。parity 阶段有 3 个词表缺失的 id（SPEED_POTION_POWER、FLEX_POTION_POWER、HATCH）。
- 下一步：先做 0b（reseed）。
- 需要 Lucien 决定的事：是否继续做 0a（内存快照）：重放只要 6–16 ms 且不是瓶颈，0a 收益有限。

### 2026-10-02 23:47 (PT) · feat/mcts-combat
- 做了什么：Step 0a（ed702b7，模拟器 d9bad0b）：内存快照，原地恢复（记录字段值，写回同一批对象），成为默认的 clone 模式；新增 census、statics、digest、dump 诊断接口。
- 结果/数据：clone 模式 1000 条随机路线全部通过（可见状态和隐藏状态摘要一致）。1 万次恢复后内存在 190–215 MB 之间，不增长。一次快照约 2.1 万个对象，捕获约 5 ms，恢复约 2.5 ms。STS2RL 测试数量未记录。
- 遇到的问题：恢复后 end_turn 卡死 15 秒（游戏线程停在另一条时间线的决策里）；装箱结构体没有记录；全局进度 DiscoveredRelics 不一致；replay 模式的隐藏状态不一致。都已修复或说明。
- 下一步：0b，reseed。
- 需要 Lucien 决定的事：无

### 2026-10-03 00:35 (PT) · feat/mcts-combat
- 做了什么：Step 0b（5113f53，模拟器 42e330e）：reseed 重设战斗 RNG 流、怪物 RNG 和抽牌堆顺序，不改可见状态；/sim/reseed，reset 带可选 reseed；STS2Client.sim_reseed。b16c7af：MCTS 记录改为每一步一份文档（docs/mcts/）。
- 结果/数据：reseed_check：clone 50/50，replay 30/30。branch_check：300 + 1000 条路线全部通过，恢复中位数约 1.3 ms。契约回归 200 局 0 错误。测试通过（614 个）。
- 遇到的问题：抽牌堆 JSON 的排序泄露抽牌顺序。模拟器资源泄漏：长时间运行后恢复从 2.5 ms 涨到 83 ms，1331 次 reset 后留下 1331 个 RunState。
- 下一步：Step 1，MCTS 搜索器。
- 需要 Lucien 决定的事：无

### 2026-10-03 08:22 (PT) · feat/mcts-combat
- 做了什么：d201ff0：词表以游戏 v0.107.1 本身为准（模拟器 --export-vocabulary，反编译 mod）。API_SUFFIXES 去掉 _POWER 后缀再查；敌人按 entity_id 去序号的部分解析；固定表补齐营火选项、地图节点、目标类型、选择界面、水晶球物品等。新增 scripts/update_vocabulary.py。
- 结果/数据：新增可解析 22 个，原来能解析的 0 个丢失，1 个改变（Monarch's Gaze 纠正）。17 局人类录像（13,285 步）词表未命中 0 种（修改前 16 种）。契约回归 parity 第一次 PASS。测试通过（620 个）。
- 遇到的问题：15 个真实存在的能力读成 <unknown>。敌人没有 id，部分名字每场不同。指纹升到 4，所有现有 checkpoint（包括 intent8h、bc_best.pt）不能加载。
- 下一步：Step 1。
- 需要 Lucien 决定的事：无

### 2026-10-03 13:20 (PT) · feat/mcts-combat
- 做了什么：Step 1-a（0b06b25）：sts2rl.search，信息集 MCTS（按动作身份建树，每次模拟恢复并 reseed，progressive widening，默认两回合深度）；LeafEvaluator；play_fight；scripts/boss_mcts_eval.py。Step 1-b（004ce9b）：用搜索过的 boss 战拟合叶子评估（逻辑回归，符号约束）；输局按造成的伤害给分；UCB 按树的值域归一化；新增 race、deck_damage、deck_block 特征。
- 结果/数据：每次模拟约 10 ms，50 次模拟每个决策约 0.5 秒。同样 357 场：手设权重 16.5% → 第一轮拟合 20.4% → 两轮拟合（符号约束）24.9%（对手设 p < 0.001）。holdout 72 场 AUC：全部决策点 0.909，第一个决策点 0.852（手设 0.812、0.820）。测试通过（1-a 后 633 个，1-b 后 638 个）。
- 遇到的问题：战斗中途的 card_select 被当成战斗结束。搜索后没有恢复到决策点。输定局面所有叶子都是 0。开局特征预测不了输赢（AUC 约 0.5）。自由拟合有几个权重符号反了。
- 下一步：Step 1-c，选搜索深度。
- 需要 Lucien 决定的事：无

### 2026-10-03 20:54 (PT) · feat/mcts-combat
- 做了什么：Step 1-c（fb07ffc）：1 / 2 / 3 回合深度，每个决策 4 秒，在 40 个快照 × 3 个世界上比较。Step 1-d（9bbec81）：143 个第 1 幕 boss 快照 × 3 个世界 = 429 场，比较 actor、MCTS-50、MCTS-200 和 clairvoyant-200（不 reseed，作为上限估计）。800 次模拟一档经讨论换成 clairvoyant。
- 结果/数据：深度 1 / 2 / 3：27.5% / 30.8% / 25.0%，每决策模拟 195 / 119 / 90 次，差异不显著（p 0.09–0.58），选 2 回合。429 场：actor argmax 6.1%，actor 采样 4.4%，MCTS-50 24.2%，MCTS-200 28.7%，clairvoyant-200 35.0%。Waterfall Giant 0–3%。测试通过（1-c 时 638 个；1-d 数量未记录）。
- 遇到的问题：1-c 第一次运行时多个线程共用函数属性，模拟次数互相覆盖，重跑。一半快照怎么打都赢不了，剩下的输局主要是牌组问题。
- 下一步：Step 2，MCTS 打全部战斗，PPO 学宏观决策。
- 需要 Lucien 决定的事：无

### 2026-10-03 20:59 (PT) · feat/mcts-combat
- 做了什么：Step 2-a/b/c（fdeeee2）：CandidatePPOAgent.choose_external 记录外部选出的动作（参与 GAE 和 critic，不参与 policy loss）；SearchCombatAgent 让搜索打战斗，其余交给 PPO；SearchDecisionRecorder 记录搜索决策和访问分布；sts2rl-train 新增 --search-combat 等选项。
- 结果/数据：测试通过（656 个）。冒烟测试 7 局（随机权重）：最高第 25、24 层，2 局打过第 1 幕 boss；742 个搜索决策；5 次更新 value_loss 0.07–0.33；约每小时 100 局。无错误、无截断。
- 遇到的问题：未记录。
- 下一步：Step 2-d 正式训练。
- 需要 Lucien 决定的事：无

### 2026-10-04 18:32 (PT) · feat/mcts-combat
- 做了什么：Step 2-d：runs/step2-search，从随机权重开始，所有战斗由 MCTS-50 打，2000 局（10-03 21:34 ~ 10-04 14:30）。34c9d0a：词表迁移脚本（intent8h → intent8h-v4）。340d5d1：resume 后搜索局编号唯一。fa526bd：修复 agent.lane 存成 LaneView 导致的崩溃。7835d8d：sts2rl-eval --search-combat。eea001f：记录。
- 结果/数据：每周期平均 floor 11.9–17.2，最好第 11 周期 17.2，最高第 48 层。记录 173,994 个搜索决策。评估（同一 MCTS-50 打战斗，150 个训练 seed）：第 11 周期 update_001052 19.41，intent8h 16.74（+2.67，p = 0.041）；最终 checkpoint 16.59（−0.15，p = 0.58）。P(跳过选牌) 在 0.00 和 1.00 之间摆动。迁移后 300 个战斗状态和 399/400 个非战斗状态输出与原模型相同。
- 遇到的问题：第 1236 局训练崩溃（lane 编号）。resume 后局编号重复。resume 时传 --seed-pool default 被拒。宏观策略不稳定：每次更新只有约 80 个 PPO 自己的决策。
- 下一步：稳定宏观训练：战斗移出 rollout，加每次更新的 KL 上限。
- 需要 Lucien 决定的事：（待讨论）按房间设置搜索的模拟次数：boss 和精英 200 次，普通 50 次，每局耗时约 1.7 倍，只能用于新 run（progress.md）。

### 2026-10-04 18:32 (PT) · feat/mcts-combat
- 做了什么：Step 3-a：Decision 可带 target_distribution，BCTrainer 用软标签交叉熵（da338d7）；scripts/search_dataset.py 把搜索记录转成数据集；boss_mcts_eval.py --bc-artifact，sts2rl-bc-train --target-temperature（4388ab5）。用 data/distill/v1-60k 蒸馏战斗 actor（runs/distill-v1）。eea001f：记录。
- 结果/数据：60,914 个决策（训练 47,068 / holdout 13,846）。保存 epoch 13：holdout 准确率 56.6%，交叉熵 1.5577；基线随机 21.5%，总选第一个 30.0%。429 场 boss 战：蒸馏 actor argmax 15.6%（holdout 快照 15.3%），采样 1.6%；intent8h actor 6.1%，MCTS-50 24.2%；对 intent8h actor 48 对 7（p = 1e-8）。训练用时 1 小时 20 分钟。测试通过（669 个）。
- 遇到的问题：50 次模拟的访问比例太平，蒸馏出的分布也平，采样时很差。全部数据放进内存需要 4–6 GB，和 7 个模拟器放不下，只用了子集。
- 下一步：T = 0.5 再蒸馏一次，argmax 和采样都评估，然后做 Step 3-b PPO 微调。
- 需要 Lucien 决定的事：无

### 2026-10-04 18:54 (PT) · feat/mcts-combat
- 做了什么：Step 2-e（f160c3e、7499cf4）：--target-kl（每个 minibatch 前估 KL，超过 1.5 倍就停止这次更新）；--search-fights-out-of-rollout（整场战斗当作一次转移，奖励并到前一个宏观决策，不跨局合并）。Step 2-f（4ebefcf）：sts2rl-surgery（migrate / outputs / compare / spec），checkpoint 自带 model_spec；sts2rl-train --init-from。
- 结果/数据：测试通过（2-e 后 681 个，2-f 后 700 个）。用新工具重做 intent8h 迁移：和 intent8h-v4 逐张量相同（151/151）。跨代码版本输出检查：499 个决策中 478 个完全相同，495 个 argmax 相同，21 个差异都来自现在能识别的 power id。update_001052 通过 --init-from 检查。
- 遇到的问题：PPO 的 clip 不保护不在 batch 里的状态（P(跳过选牌) 0.21 → 1.00）。第一次迁移用错旧 spec（715b65b），被形状检查拒绝；正确的是 4e1a30f。
- 下一步：从 update_001052 用 --init-from，加两个稳定措施，重新训练宏观策略。
- 需要 Lucien 决定的事：无

### 2026-10-05 03:09 (PT) · feat/mcts-combat
- 做了什么：Step 2-g：runs/step2e-stable，从 update_001052 开始，--search-fights-out-of-rollout，--target-kl 0.02，600 局，172 次更新（10-04 18:55 ~ 10-05 03:02，中间暂停一次）。bab2de1：修复 surgery 对相对路径的 run 目录多拼一层 checkpoints/。da9db86：记录。
- 结果/数据：周期平均 floor 18.4 / 18.8 / 19.3 / 19.6，都高于 Step 2-d 的所有周期。对起点同 seed 16.95 → 18.99（+2.0，261 高 / 163 低，p < 0.005）。172 次更新中 171 次 KL 早停，平均 7.5 步。value loss 平均 0.34。第 1 幕 boss 胜率：Ceremonial Beast 71%，Soul Fysh 64%，Vantom 52%，Lagavulin Matriarch 48%，The Kin 39%，Waterfall Giant 19%。P(跳过选牌) 全部为 0.00。P(休息) 0.58 → 1.00，与 HP 无关。截断 2 局。测试通过（704 个）。
- 遇到的问题：满血也休息，放弃升级。报告脚本在 resume 后重复计算局数。surgery 相对路径 bug。
- 下一步：加当前幕 boss 特征（act_boss），surgery 迁移后继续训练。
- 需要 Lucien 决定的事：给休息选项加"实际恢复量"一列：按指示暂不加，留待以后决定。

### 2026-10-05 17:33 (PT) · feat/mcts-combat
- 做了什么：Step 2-h（7765a19）：mod（feat/run-boss，0318851）和模拟器（38fc195）在 run 块里加 run.boss；schema 加 act_boss 列；surgery 迁移 step2e-stable 的 update_000172，开始 runs/step2h-boss。Step 2-i（feae10e）：--search-elite-simulations、--search-boss-simulations；runs/step2i-mcts200 精英和 boss 用 200 次模拟。5e567a3：记录。
- 结果/数据：step2h-boss 656 局，193 次更新：周期 19.7 / 20.6 / 20.0 / 18.8 / 19.5（部分），平台期约 19.7。换不同 boss，选牌概率最多变约 10%。step2i-mcts200 178 局：第 1 周期 20.3，和 boss run 同 seed +0.8（p = 0.67），每局时间 2.2 倍。测试数量未记录。
- 遇到的问题：模型能看到 boss，但还没学会按 boss 构筑牌组。200 次模拟按时间算不划算，停止。同时发现 Waterfall Giant 的评估错误。
- 下一步：先修 Waterfall Giant 的评估错误（Step 2-j）。
- 需要 Lucien 决定的事：mod 的 run.boss 改动还没有在真实游戏中测试（需要登录 Steam）。

### 2026-10-05 17:33 (PT) · feat/mcts-combat
- 做了什么：Step 2-j：70068df：被打死的 Waterfall Giant（HP 999,999,999）按"正在死"处理，Steam Eruption 计入来袭伤害。ef26810：--init-optimizer 带上 Adam 的矩。78ab22c：boss_mcts_eval --only。用回归分析 HP、回合数和牌组。runs/step2j-fixed 带修复训练。5e567a3：记录。
- 结果/数据：298 场输掉的 Giant 战中 176 场 Giant HP 低于 15%。66 个 Giant 快照：1/66 → 3/66；429 个快照 104 → 106。训练中 Giant 胜率约 14% → 33–50%。回归 R²（加入牌组）：每回合伤害 0.15 → 0.47，战斗余量 0.19 → 0.42。人类对 agent：过第 1 幕 boss 17/17 对 55%，升级 3.8 对 0.9，药水 2.5 对 0.7，牌组 17.1 对 19.8。约 40% 的 boss 战带着 150 以上没花的金币。step2j-fixed 周期 17.5 / 19.2。测试数量未记录。
- 遇到的问题：搜索避开最后一击。--init-from 不带 Adam 矩，前几次更新选牌 KL 约 3 倍，第 1 周期比 boss run 低约 2 层。
- 下一步：从 boss run 平台期用 --init-optimizer 重新开始（Step 2-k）。
- 需要 Lucien 决定的事：无

### 2026-10-05 19:48 (PT) · feat/mcts-combat
- 做了什么：Step 2-k：runs/step2k-plateau 从 step2h-boss/update_000192 用 --init-optimizer 继续；冻结对照 runs/step2k-frozen（学习率 1e-12）。Step 2-l：71042a9：敌人剩余 HP 按搜索根状态计算，计入死亡召唤；27f52d5：手里不能打出的牌的回合末伤害计入来袭伤害；A/B 对照 runs/step2l-frozen-fix。0469f18：记录。
- 结果/数据：2-k：训练 184 局 18.3，冻结 150 局 18.1；同 150 个 seed 18.33 对 18.09（p = 0.21），下降不是 PPO 更新造成的，起点本身约 18.1–18.3。723 局中 207 局（29%）死在第 17 层前；Phrog Parasite 死亡率 81%。2-l A/B（同一冻结策略）：18.09 → 19.98（+1.89，78 高 / 43 低，p ≈ 0.002）；第 9 层前死亡 23% → 12%；第 1 幕精英死亡率 33% → 19%，普通战斗 2.9% → 1.5%；boss 快照 106 → 106。MCTS-200 对 MCTS-50：127 对 106（p = 0.003）。测试通过（718 个）。
- 遇到的问题：基线错误：上一个 run 的周期均值包含更早更好的阶段，偏高。击杀后敌人剩余比例反而上升（2366 次击杀中 57%），评估分数下降（30%）。Phrog Parasite 和 Gremlin Merc 的死亡召唤让"几乎打死"比"打死"分数高。
- 下一步：带两个修复的新训练 run，从 update_000192 用 --init-from --init-optimizer 开始；之后比较都用同一 checkpoint 的冻结对照。
- 需要 Lucien 决定的事：无

### 2026-10-06 07:05 (PT) · feat/mcts-combat
- 做了什么：Step 2-n（b3a7edb）：runs/step2n-train，从 step2h-boss/update_000192 开始，带评估修复，boss 战 200 次模拟，600 局，202 次更新（10-05 20:03 ~ 10-06 07:02）。
- 结果/数据：周期平均 floor 21.95 / 23.25 / 22.37 / 22.01；最高第 48 层，通关 0。对冻结基线 +2.05 / +3.35 / +2.27 / +1.97。第 1 幕 boss 通过率 68% → 60%。P(拿遗物) 1.00 → 0.02（update 144）；第 1 幕 boss 前遗物 4.1 → 3.2。第 2 幕 boss 只赢 20–33%。截断 2 局，action error 0。
- 遇到的问题：第 2 周期之后不再提高。策略学会了在宝箱房不拿遗物（proceed 少一步，省 0.01）。升级 0.6 张、药水 0.6 瓶，牌组构筑没有改善。
- 下一步：修复宝箱（先拿遗物再离开）；修复后从本 run 的 checkpoint 继续，基线用同一 checkpoint 的冻结 run。
- 需要 Lucien 决定的事：牌组构筑的方案：人类录像行为克隆加 KL 约束（提案 C）文档列为候选，未决定。

### 2026-10-06 12:56 (PT) · feat/mcts-combat
- 做了什么：Step 2-o（fd4e3c4）：_treasure_actions 和奖励界面同规则：有遗物时只提供 claim_treasure_relic，拿完才提供 proceed。Codex 审查设计和 diff。重新清洗 BC 数据集 data/bc/v7。按要求删除 3 条 schema 1 的录像。
- 结果/数据：人类记录 65 次宝箱访问，61 次拿取后宝箱清空；模拟器 2 个 seed 实测拿取后 relics 为 []。测试通过（735 个）。data/bc/v7：10771 个决策（v6 为 10824），强制步 2134，不匹配 250。
- 遇到的问题：data/bc/v6 的 53 个宝箱决策和新动作空间不一致，被 BCDataset 拒绝。
- 下一步：在新分支 feat/pbrs 上测试基于势函数的奖励塑形（PBRS），解决营火总是休息。
- 需要 Lucien 决定的事：无

### 2026-10-06 13:27 (PT) · feat/mcts-combat
- 做了什么：Step 2-p 修复（e0387f2）：hold_open_steps：每个 lane 最后一个未完成的决策留到下一次更新；update() 只训练已完成的前缀，用被保留记录的采样价值 bootstrap；自动更新在持锁时复查阈值。Codex 审查两次。
- 结果/数据：估算（未实测）：约 14% 的战斗跨越一次更新，600 局中约 800–1000 个决策受影响。关闭开关时结果与修改前相同。测试通过（756 个）。
- 遇到的问题：Codex 审查 PBRS 设计时发现：之前所有用 --search-fights-out-of-rollout 的 run（2-e 到 2-n）里，正在打战斗的 lane 的决策在战斗结束前被训练，战斗中的死亡可能没有记为终局。checkpoint 不保存被保留的记录。
- 下一步：从 step2n 的 update_000102 继续训练 450 局，10 个模拟器。
- 需要 Lucien 决定的事：无

### 2026-10-06 15:06 (PT) · perf/fast-client
- 做了什么：Step 2-p 性能：d43fcfc：HTTP 客户端 trust_env = False。55f36c0、6e112af：STS2Client 改用 urllib3 连接池；restore 和 reseed 合并成一个请求（模拟器 785bf21）。a61574d：记录。之后快进到 feat/mcts-combat。
- 结果/数据：10 个模拟器时约 53 局/小时，和 7 个相同；每个 lane 每步 1.86 → 2.82 秒。py-spy：.netrc 查找约 12%，注册表代理约 14%，单独的 restore 和 reseed 约 20%。bench：192 → 316 周期/秒（+64%），客户端 CPU 3.05 → 1.02 ms/周期。合并请求等价性：clone 84/84，replay 42/42。测试通过（792 个）。
- 遇到的问题：瓶颈是训练进程，不是模拟器。第一次启动的 runs/step2p-train 在 63 局时随会话重启停止，不计入结果。
- 下一步：用同样设置在 runs/step2p-fast 重新训练。
- 需要 Lucien 决定的事：无

### 2026-10-06 20:17 (PT) · feat/mcts-combat
- 做了什么：Step 2-p 训练：runs/step2p-fast，从 step2n-train/update_000102 用 --init-from --init-optimizer 开始，带宝箱修复、奖励归属修复和性能优化，10 个模拟器，450 局，164 次更新（10-06 15:06 ~ 20:11，exit 0）。a85c071：记录（docs/mcts/step2p-training.md）。
- 结果/数据：周期平均 floor 24.75 / 22.88 / 24.25。第 1 周期对 Step 2-n 第 2 周期 +1.24（p = 0.10），对冻结基线 +4.82（p < 0.0001）；第 2 周期对第 1 周期 −2.22（p = 0.035）。到第 3 幕 boss 8 局，通关 0。第 2 幕 boss 死亡率 74–79%。第 1 幕 boss 前遗物回到约 4.2，升级牌少于 1 张。约 88 局/小时（Step 2-n 约 55）。截断 2 局（商店 Foul Potion），action error 0。
- 遇到的问题：继续训练没有让策略持续提高，最好的周期离起点最近。周期之间波动约 2 层。宏观策略没有学会构筑牌组（升级、药水、金币）。
- 下一步：用冻结策略在同一批 seed 上评估 update_000060、update_000108、update_000164，选出最好的起点；分析第 2 幕 boss 战；牌组构筑需要更强的信号。
- 需要 Lucien 决定的事：无

### 2026-10-06 20:17 (PT) · fix/refused-actions
- 做了什么：Step 2-q：f389cbe：先临时去掉 (FOUL_POTION, shop)；runner 不在没变化的画面上重复被拒的动作。f5b1474（审查后）：同一动作连续被拒 2 次才排除；搜索根节点先过滤再分组。a10909b：两个后端都支持后，恢复 (FOUL_POTION, shop)。模拟器 d985824（Harmony 补丁），mod 09c4938、fa284de（feat/shop-foul-potion，使用前关闭商人物品栏）。a85c071：记录（docs/mcts/step2q-foul-potion.md）。
- 结果/数据：runs/step2* 的 5040 局中 14 局截断，全部是商店里的 Foul Potion 被拒；13 次拒绝中 12 次是连续重复。测试通过（813 个）。模拟器已重新编译。
- 遇到的问题：mod 读商店状态时会打开物品栏，模拟器没有商人界面，两个后端都拒绝这个药水。第 1 次拒绝就排除会把时序竞争变成不可撤回的选择。搜索根节点先分组再过滤，被拒的副本仍是代表。
- 下一步：mod 部署之前，真实游戏仍会拒绝这个药水；runner 在 2 次拒绝后排除它，不会截断。
- 需要 Lucien 决定的事：mod 改动（feat/shop-foul-potion，09c4938、fa284de）已编译，未部署，未推送。需要按 docs/testing/foul-potion-shop.md（mod 仓库）在真实游戏中测试，再决定部署。

### 2026-10-07 03:39 (PT) · feat/mcts-combat
- 做了什么：画 step2p 第 3 周期的楼层分布图，写阶段文档 step2p-training.md 和 step2q-foul-potion.md（a85c071）。sts2rl-eval 每完成一局输出一行 JSON（1be5bd5）；新增 --holdout-seeds，可以给没有记录 holdout 的 checkpoint 指定 holdout seed，和训练池重叠时拒绝（7c3f65a）。开始 holdout 评估（runs/eval2p_holdout_driver.sh）：step2n update_000102、step2p update_000060 / 000108 / 000164，30 个 holdout seed，每个 seed 3 局，宏观决策用 argmax，战斗 MCTS 50 次、boss 200 次。在 CLAUDE.md 加入开发日志规则，新建本文件。
- 结果/数据：holdout 评估未完成。测试通过（816 个），ruff 无问题。
- 遇到的问题：先在 150 个训练 seed 上开始了冻结策略评估，按要求改为 holdout 后停止（约 10 局，不使用）。step2n/step2p 训练时没有传 --holdout-seeds，checkpoint 没有记录 holdout，sts2rl-eval 拒绝运行，所以加了 --holdout-seeds。driver 脚本的退出码记录有错（$(date) 在 $? 之前执行，所以总是 0），新 driver 已改。
- 下一步：holdout 评估完成后按 seed 配对比较 4 个 checkpoint，选出起点，继续训练。
- 需要 Lucien 决定的事：mod 的 Foul Potion 改动需要在真实游戏中测试后再部署；是否合并到 main。

### 2026-10-07 05:58 (PT) · feat/mcts-combat
- 做了什么：Step 2-r：在 30 个 holdout seed 上（每个 seed 3 局，argmax，MCTS 50 / boss 200）用冻结策略评估 step2n update_000102 和 step2p update_000060 / 000108 / 000164（runs/eval2p-holdout，20:39 ~ 22:55）。用 scripts/policy_probe.py 在人类录像的 1317 个宏观界面上探测 15 个 checkpoint 的动作概率。写 docs/mcts/step2r-holdout-eval.md；scripts/policy_probe.py、scripts/holdout_compare.py 入库。
- 结果/数据：平均楼层 n102 23.92、p060 26.30、p108 25.42、p164 22.32（各 90 局）。同 seed 配对：p060 对 n102 +2.38（17 高 / 6 低，p = 0.035）；p164 对 p060 −3.98（6 高 / 20 低，p = 0.009）；p108 对 p060 −0.88（p = 0.69）。通关 0。探测：营火升级概率 0.01–0.04（人类 0.65），跳过选牌 0.00（人类 0.52），HP > 80% 时休息 0.83–0.94（人类 0.06），整个训练过程都没有变化。测试通过（816 个）。
- 遇到的问题：22:43 模拟器被误停（等待脚本匹配了日志里上一次启动留下的 "all done" 行），n102 和 p060 各损失 2 和 6 局，已按 seed 重放补齐。BC 参考策略训练和 Codex 的 BC-KL 设计审查被 Claude Code 因内存不足停止，之后不再需要。
- 下一步：从 update_000060 开始，加定向探索（营火 0.3、选牌奖励 0.3、地图 0.15），训练 3 个周期；每个周期末在 holdout 上评估并探测升级、跳过的概率。设计已交 Codex 审查（runs/codex/exploration_design.md）。
- 需要 Lucien 决定的事：无（定向探索方案已由 Lucien 选定；mod 的 Foul Potion 改动仍待真实游戏测试）。

### 2026-10-07 11:31 (PT) · feat/mcts-combat
- 做了什么：Step 2-s：实现定向 ε 探索（85c6703，Codex 审查设计和 diff；混合分布作为被训练的策略），从 step2p update_000060 训练 3 个周期（runs/step2s-explore，23:36 ~ 04:28，450 局，153 次更新，--explore rest_site=0.3,card_reward=0.3,map=0.15）。每个周期末探测动作概率，在 holdout 上评估。写 docs/mcts/step2s-exploration.md。另外实现了 BC-KL 参考项（feat/bc-kl，b1f6a29 + 0c68d21，Codex 审查两轮），已合并进 feat/mcts-combat。
- 结果/数据：周期平均楼层 23.13 / 22.04 / 22.11；同 seed 对 Step 2-p 三周期平均 −0.64 / −1.83 / −1.85（p = 0.55 / 0.060 / 0.005）。到第 3 幕 boss 6 局，通关 0。探测：升级概率 0.02 / 0.00 / 0.01（起点 0.01，人类 0.65），跳过选牌 0.00 全程（人类 0.52）。holdout：周期 1 末 update_000054 对 p060 −3.03（8 高 / 16 低，p = 0.15，79 局）；周期 2、3 末未完成。测试：探索 862 个通过，BC-KL 918 个通过。
- 遇到的问题：探索没有恢复塌缩的选项，策略反而变差，商店漂移到买药水、丢药水。周期 1 的 holdout 评估在 79 局时因模拟器故障（水晶球事件后 run faulted）退出。顺序执行的周期评估脚本会把"最新"checkpoint 当成周期末，已改为手动指定。
- 下一步：Step 2-t：从 p060 开始，--reference-policy runs/bc-v7/bc_best.pt --reference-kl 0.1，不探索，3 个周期；参考模型正在训练（runs/bc-v7）；周期 2、3 的 holdout 评估进行中，完成后补进文档。
- 需要 Lucien 决定的事：无（mod 的 Foul Potion 改动仍待真实游戏测试）。

### 2026-10-07 12:39 (PT) · feat/mcts-combat
- 做了什么：补齐 Step 2-s 周期 2、3 末 checkpoint 的 holdout 评估（各 90 局），更新 docs/mcts/step2s-exploration.md。训练 BC 参考模型 runs/bc-v7（data/bc/v7 全部 10771 个决策，04:29 ~ 05:05）。启动 Step 2-t（runs/step2t-bckl）：从 step2p update_000060 开始，--reference-policy runs/bc-v7/bc_best.pt --reference-kl 0.1，不探索，10 个模拟器，450 局；周期评估脚本改为"到周期末立即记录并探测，评估按队列顺序"。
- 结果/数据：holdout 对 p060（同 30 seed）：周期 1 末 −3.03（p = 0.15），周期 2 末 −1.77（p = 0.52），周期 3 末 −0.08（10 高 / 16 低，p = 0.33）；周期 3 末对 p164 +3.46（p = 0.036）。bc-v7：选第 4 个 epoch，留出交叉熵 1.117，准确率 56.4%（营火 76.5%，选牌 65%，地图 75.5%）；参考策略在人类界面上升级 0.68、跳过选牌 0.62。Step 2-t 未完成。
- 遇到的问题：无。
- 下一步：Step 2-t 每个周期末探测升级、跳过的概率，在 holdout 上评估；结束后写文档。
- 需要 Lucien 决定的事：无。

### 2026-10-07 19:13 (PT) · feat/mcts-combat
- 做了什么：Step 2-t（全局 β = 0.1，runs/step2t-bckl，05:38 ~ 06:49，151 局后停）和 Step 2-u（按界面 β：营火 0.1、地图 0.03，runs/step2u-bckl-screens，06:49 ~ 12:09，450 局，163 次更新）。按界面系数的实现 d1d2622 + 7a5db25（Codex 审查，951 个测试通过）已合并 push。每个周期末探测、holdout 评估。写 docs/mcts/step2tu-reference-kl.md。
- 结果/数据：Step 2-t 第 1 周期平均 17.92，holdout 对 p060 −5.96（p = 0.002）；选牌的熵 0.41 → 1.27。Step 2-u 周期平均 24.56 / 23.38 / 22.82；同 seed 对 Step 2-p 第 1 周期 +0.52（p = 0.80），全程约 −0.3（不显著）；营火升级概率 0.01 → 0.26 / 0.27 / 0.20；进入第 1 幕 boss 时升级牌 1.32 / 1.29 / 1.27（Step 2-p 0.66–0.83），HP 90% → 83%；holdout 周期 1 末 24.90，对 p060 −0.96（p = 0.84）；周期 2、3 末未完成。通关 0。0 错误。
- 遇到的问题：全局 β 复制了参考模型在选牌、商店上的不确定，策略变成随机；按界面之后解决。升级概率在 0.26 附近停住（β = 0.1 和 critic 的休息偏好平衡），少休息让 boss 前 HP 下降，楼层持平。停止时的 approx KL 偶尔到 1.2，是采样估计量被参考项抬高的稀有动作推高的。
- 下一步：等周期 2、3 的 holdout；决定下一个 run：营火 β 0.3，或从 p060 跑 6 个周期的普通 PPO 对照。
- 需要 Lucien 决定的事：下一个 run 的方向（见上）；mod 的 Foul Potion 改动仍待真实游戏测试。

### 2026-10-07 23:02 (PT) · feat/mcts-combat
- 做了什么：Step 2-v：从 step2p update_000060 跑 6 个周期的普通 PPO（runs/step2v-plain6，12:19 ~ 22:58，900 局，331 次更新，没有探索和参考 KL），每个周期末探测、holdout 评估、画图。写 docs/mcts/step2v-plain6.md。
- 结果/数据：周期平均楼层 23.33 / 24.23 / 25.31 / 24.15 / 25.39 / 23.77；全部 900 局对 Step 2-p 三周期平均 +0.38（441 高 / 401 低，p = 0.18）；第 6 周期对第 5 周期 −1.76（p = 0.037）。holdout 对 p060：周期 1–4 末 −0.67 / −2.28 / −0.14 / −1.84（p 0.69 / 0.23 / 1.0 / 0.69），周期 5、6 末未完成。进入第 3 幕 boss 16 次，全部失败，通关 0。探测：升级 0.00–0.03、跳过 0.00–0.01 全程不变。0 错误。测试 951 个通过（代码未改）。
- 遇到的问题：无新问题。一个旧的等待脚本被 Claude Code 因内存不足停止，无影响。
- 下一步：等周期 5、6 的 holdout 评估，补进文档。下一个 run 待定：营火 β 0.3，或先分析第 2、3 幕 boss 战的失败原因。
- 需要 Lucien 决定的事：下一个 run 的方向；mod 的 Foul Potion 改动仍待真实游戏测试。

### 2026-10-08 19:10 (PT) · feat/mcts-combat
- 做了什么：Step 2-w：从 Step 2-v 末尾（update_000331，带优化器）继续普通 PPO，先定 10 个周期，10:30 按 Lucien 的决定扩到 20 个周期（1500 局后 `--resume --total-episodes 3000`，12:51 接上）。训练中不做 holdout 评估，每周期末记录 checkpoint + 探测 + 画图。19:05 按 Lucien 的指令暂停：训练、watcher、10 个模拟器全部关闭，等指令再启动。Step 2-v 的周期 5/6 holdout 评估也已停止（59/90、31/90，后者为模拟器故障），文档里标为未完成。
- 结果/数据：2125 局（14 个完整周期 + 25 局），最后 checkpoint `update_000718.pt`。周期均值 23.41 / 24.35 / 22.85 / 20.79 / 19.90 / 22.04 / 22.76 / 21.89 / 22.37 / 22.36 / 22.19 / 23.35 / 23.09 / 23.62。2100 局对 2-v 六周期均值（同 seed）−1.86（729 高 / 1322 低，p < 0.001），对 2-p −1.48。探测：营火升级在周期 1 末 0.13 → 周期 2 末 0.04 → 周期 3 末 0.68 → 周期 4–5 末 0.90–0.92（入场升级牌 3.7 张、入场 HP 55%、第 1 幕 boss 死亡率 62%，周期均值跌到 19.9）→ 周期 6 起回到 0.00–0.02（休息 0.93–1.00）。跳过选牌全程 ≤ 0.04。第 3 幕 boss 14 次（全部在周期 1–5），0 胜；通关 0。0 截断，0 错误。文档未写（未完成）。
- 遇到的问题：Claude Code 因内存不足回收了 4 次等待脚本（空闲内存在探测时短暂 < 2.5 GB），训练未受影响；停掉 2 个空闲模拟器（15612、15613）后缓解。第 11–20 周期的 watcher 第一版把接续脚本日志里的 "first driver … exit 0" 当成训练结束而退出，已修复重起。第一次起训练误用了 WSL 的 bash.exe，立即失败，换 Git Bash 后正常。
- 下一步：等 Lucien 指令。恢复命令：`uv run sts2rl-train --run-dir runs/step2w-plain10 --resume --total-episodes 3000`（先起 10 个模拟器 15600–15609）；或者就此收尾：写 docs/mcts/step2w-plain20.md，评估 p060（重评）、2-v 末、2-w 周期 2 末和最后一个 checkpoint 的 holdout。
- 需要 Lucien 决定的事：Step 2-w 是继续到 20 个周期还是就此收尾；mod 的 Foul Potion 改动仍待真实游戏测试。

### 2026-10-09 02:50 (PT) · feat/mcts-combat
- 做了什么：按 Lucien 的指令恢复 Step 2-w（01:04，`--resume --total-episodes 3000`，从 `update_000718`、2124 局继续，10 个模拟器，新驱动日志 `runs/step2w-plain10.driver3.log`，周期记录脚本 `runs/step2w_cycle_record3.sh` 按最大局号判断周期边界）。第 2210 局通关，按指令保存 seed 和检查点：`runs/wins/JEQSXL4XVT/`（`update_000742/748/754`、这一局的 335 个战斗决策、`info.json`），记录写进 docs/mcts/wins.md。
- 结果/数据：第一次通关：seed `JEQSXL4XVT`，第 2210 局，奖励 71.23，楼层 48，577 步，boss Soul Fysh → The Insatiable → Aeonglass（进入时 HP 53/82、27/82、21/82）。恢复后到第 2245 局：121 局，平均楼层 25.06，最高 50（第 2138 局，seed `6S8W3SPCTK`，死在第 3 幕 boss）。周期 15 前 140 局平均 24.69，同 seed 对周期 14 +1.17（p = 0.58）。截断 0，错误 0。
- 遇到的问题：无。
- 下一步：Step 2-w 继续到 3000 局（周期 15–20，约 11 小时，按约 75 局/小时）；结束后写 docs/mcts/step2w-plain20.md。
- 需要 Lucien 决定的事：无。
