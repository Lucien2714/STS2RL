# 无渲染运行游戏客户端

STS2RL 的训练需要真实的游戏客户端。本文档说明如何以无窗口（headless）方式运行
这些客户端，以及多开时必须处理的存档隔离和帧率设置。

游戏引擎是 Godot 的定制构建（MegaDot v4.5.1），`--headless` 是引擎自带参数。
经实测，游戏逻辑、场景树、STS2MCP mod 与 HTTP API 在无渲染器时全部正常工作。

## 用法

```powershell
# 4 个客户端，无窗口，每个客户端独立存档
pwsh -File scripts/start_game_clients.ps1 -Count 4 -Headless

# 只检查配置，不启动任何进程
pwsh -File scripts/start_game_clients.ps1 -Count 4 -Headless -ValidateOnly

# 只启动指定客户端
pwsh -File scripts/start_game_clients.ps1 -Indices 2,3 -Headless
```

脚本持续守护这些进程：每 `-CheckIntervalSeconds` 秒调用一次各客户端的 API，
连续失败 `-MaxFailedHealthChecks` 次即重启该客户端；进程自行退出时同样重启。
Ctrl+C 关闭全部客户端。

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-Count` | `4` | 启动的客户端数量 |
| `-Indices` | 无 | 只启动指定序号的客户端，覆盖 `-Count` |
| `-GameRoot` | `D:\sts2depot` | 各客户端游戏安装所在目录 |
| `-InstanceFormat` | `sts2_{0}` | 安装目录名模板，`{0}` 为 1 起的客户端序号 |
| `-BasePort` | `15526` | 客户端 1 的端口，其余依次递增 |
| `-Headless` | 关 | 无窗口运行 |
| `-SharedSaves` | 关 | 关闭存档隔离，回到旧批处理脚本的行为 |
| `-SaveRoot` | `%LOCALAPPDATA%\STS2RL\clients` | 各客户端存档目录的父目录 |
| `-SeedFrom` | `%APPDATA%\SlayTheSpire2` | 播种 `settings.save` 的来源 |
| `-ValidateOnly` | 关 | 只校验并打印计划，不启动 |

本脚本取代仓库根目录下的 `start_three_game_clients.bat` 与
`start_four_game_clients.bat`。那两个文件内容相同，仅实例数不同，且其中每一项
操作都已经在调用 PowerShell。

## 无渲染模式关闭了什么

`--headless` 等价于 `--display-driver headless --audio-driver Dummy`。脚本将其
展开书写，以便日后可分别排查两者。

有窗口运行时日志开头包含渲染设备：

```
MegaDot v4.5.1.m.12.mono.custom_build
D3D12 12_0 - Forward+ - Using Device #0: NVIDIA - NVIDIA GeForce RTX 5070
WARNING: PSO caching is not implemented yet in the Direct3D 12 driver.
```

无渲染运行时该设备不再创建，资源统计中显存为零，而 mod 正常加载：

```
MegaDot v4.5.1.m.12.mono.custom_build
FMOD Sound System: Successfully initialized
[INFO] [Startup] Resource stats (main menu loaded (complete)): StaticMem=0B, VRAM=0B, ...
[STS2 MCP] v0.5.0 server started on http://localhost:15527/
```

FMOD 音频与 Steam 在无渲染模式下均初始化成功。

## mod 不依赖渲染器

Godot 的 headless 模式移除渲染，但保留场景树。控件的可见性、布局与矩形范围由
控件自身代码计算，与是否存在渲染器无关。

STS2MCP 只依赖后者。对 `STS2_MCP.dll` 全部字符串的扫描结果为：

| 符号 | 引用次数 |
|---|---|
| `DisplayServer` | 0 |
| `RenderingServer` | 0 |
| `Texture` / `Viewport` / `Image` / `Sprite` | 0 |
| `WarpMouse` | 0 |
| `ParseInputEvent` / `PushInput` | 0 |

mod 触发按钮使用 `ForceClick` 与 `EmitSignal`，即直接调用控件自身的处理函数，
不经过操作系统鼠标或显示服务。该路径在无窗口时同样成立。

## 必须关闭后台限帧

`settings.save` 中的 `limit_fps_in_background` 决定游戏在非前台时是否降低帧率。
无渲染客户端永远不是前台窗口，有窗口的多客户端也只有一个处于前台，因此该设置
默认对几乎所有训练客户端生效。

这直接影响吞吐量：mod 的状态结算逻辑等待游戏处理帧，帧率下降即步速下降。

同一客户端安装、同一 seed 池、每组 4 局约 300 步的实测结果：

| 配置 | 秒/步 | 步数 / 耗时 | truncated |
|---|---|---|---|
| 无渲染 + 关闭后台限帧 | **0.524** | 300 步 / 157 秒 | 0/4 |
| 有窗口（默认设置） | 0.608 | 310 步 / 188 秒 | 0/4 |
| 无渲染 + 默认设置 | 1.166 | 312 步 / 364 秒 | 0/4 |

差异来自限帧设置，而非缺少渲染器。脚本在播种存档时自动将该项置为 `false`，
只修改客户端自己的副本，不写入用于手动游玩的真实存档。

同一结论适用于有窗口的多客户端配置：非前台的那几个同样被限帧。

## 每客户端的资源占用

单客户端稳定运行时的实测值：

| 指标 | 有窗口 | 无渲染 |
|---|---|---|
| 工作集 | 829 MB | 745 MB |
| 私有内存 | 1854 MB | 725 MB |
| 独占显存 | 1116 MB | 0 MB |
| 线程数 | 64 | 39 |

本机为 15.1 GB 内存、12 GB 显存。**限制客户端数量的是内存而非显存**：按私有
内存计算，有窗口每客户端约 1.8 GB，无渲染约 0.7 GB，后者可容纳约 2.5 倍数量的
客户端。

## 每客户端必须有独立存档目录

原批处理脚本让所有客户端共用一个存档目录。日志中的表现：

```
3907 次  [WARN] Cloud write failed for modded/profile1\saves\progress.save
  28 次  [WARN] Rename failed (attempt 3/4) …current_run.save.tmp
   7 次  [WARN] Failed to finalize backup for …progress.save
```

多个进程写入同一批文件并互相覆盖。存档损坏会使 `ResetController` 进入非预期的
菜单状态，而菜单竞态是已知的长跑杀手（见 CLAUDE.md 中 373 局中断的记录）。

### 隔离方式

该 Godot 构建通过环境变量 `APPDATA` 解析 `user://` 路径。二进制中存在 `APPDATA`
与 `LOCALAPPDATA` 的引用，Windows 构建中不含 `XDG_CONFIG_HOME`。

引擎没有 `--user-dir` 参数，`use_custom_user_dir` 是打包进 `.pck` 的项目设置，
无法在运行时更改。因此环境变量是唯一可用手段。

脚本为每个客户端设置独立的 `APPDATA`：

```
%LOCALAPPDATA%\STS2RL\clients\client-N\Roaming\
```

`-SharedSaves` 可恢复共用行为，仅用于复现旧配置。

### settings.save 必须播种

全新的存档目录会导致 **mod 完全不加载**。表现为游戏正常启动并进入主菜单，
但 API 端口始终拒绝连接，日志中出现：

```
[INFO] Skipping loading mod STS2_MCP, user has not yet seen the mods warning
```

「已确认 mod 风险提示」这一状态记录在 `settings.save` 中，而该文件不参与 Steam
云同步——存档与历史记录会同步下来，它不会。因此全新目录得到的是一份默认
`settings.save`，其中 `mods_enabled` 为 false。

脚本在首次创建客户端目录时，将 `-SeedFrom` 下的所有 `settings.save` 按相同相对
路径复制过去，覆盖 Steam 账号目录与非 Steam 的 `default` 目录。该文件携带
`mods_enabled`、`mod_list` 与 `seen_ea_disclaimer`。

**只复制这一个文件是刻意的。** 其余存档要么由 Steam 同步，要么按需生成；存档
隔离的目的正是让各客户端独立演进，复制更多会使它们从同一进度开始。

## 端口来自 mod 配置文件

各客户端的端口写在其安装目录下的 `mods/STS2_MCP/STS2_MCP.conf`：

```json
{ "port": 15527, "instant_mode": true }
```

mod 不接受命令行端口参数，修改后需重启游戏。

脚本在启动前读取该文件并与预期端口（`-BasePort` 加序号）比对，不一致即报错退出。
静默的端口错配意味着训练器连接到了非预期的客户端，该 lane 的全部指标失效，
而不会产生任何错误。

由于端口位于配置文件中，**每个客户端需要独立的游戏安装**；同一安装的两个进程会
争用同一端口。`TrainingConfig` 同样拒绝重复端口。

## 故障排查

日志位于 `logs/game_clients/client-N.stdout.log` 与 `.stderr.log`。

| 现象 | 原因与处理 |
|---|---|
| 启动正常但 API 拒绝连接，日志含 `Skipping loading mod` | 存档目录未播种 `settings.save`，见上文 |
| 启动即崩溃，栈中出现 `RenderingServer` | 某启动路径依赖渲染器。改用 `-ExtraGameArgs '--rendering-driver','dummy'` 保留窗口 |
| 日志停在 FMOD 初始化之前 | 音频设备问题。无声卡的机器需单独处理 |
| 脚本报端口不一致 | 修改对应安装的 `STS2_MCP.conf`，或传入匹配的 `-BasePort` |
| 步速明显低于上表 | 检查 `limit_fps_in_background` 是否为 `false` |
| `truncated` 比例上升 | 见下节 |

## truncated 是最该监控的指标

任何针对客户端的改动——无渲染、增加实例数、调整 `--action-delay`——都必须观察
截断率，而不只是 floor。

一局被截断时游戏中的 run 仍然存活。下一局 reset 在 `--allow-active-run` 下会
接管该 run，而不是开始它本应运行的 seed。`reused_run` 随之上升，seeded 实验就
不再测量它声称测量的内容，且全过程不产生任何错误。

```bash
uv run python scripts/analyze_run.py runs/<实验目录>
```

输出的第三行同时给出吞吐量与截断率：

```
throughput: 0.524 s/step (300 steps in 157s)   truncated: 0/4 (0.0%)
```

**步速更快但截断更多的配置并不更快。**

## 致谢

- **Mega Crit Games** — Slay the Spire 2。
- **kunology** — STS2MCP mod 的原作者。本仓库使用的是 Lucien2714 基于其工作
  修改的版本，见 mod 的 `mod_manifest.json`。
- **spire-codex**（<https://github.com/ptrlrd/spire-codex>）— `src/sts2rl/data/json/`
  下游戏数据表的来源。

## 相关文档

- [环境](environment.md) — `GameEnv` 的行为契约
- [训练](training.md) — 检查点与恢复
- [STS2MCP API 参考](raw-full.md) — 权威接口契约
