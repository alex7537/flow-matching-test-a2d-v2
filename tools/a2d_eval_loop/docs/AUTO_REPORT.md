# 自动测试报告与 Obsidian

`watch_eval_to_obsidian.py` 独立观察已有测试，每分钟从终态 JSONL 更新本地和 Obsidian 报告。完成、暂停或中断均保留真实覆盖和分母；不控制模型、不重发动作、不启动下一轮训练、不发布远端内容。

## 与现有组件的分工

| 组件 | 职责 |
|---|---|
| 已有 orchestrate / supervise / evaluate | 启动仿真、调度回合、恢复异常、保存逐轮结果；仍由外部已有 runner 提供 |
| loop.py / watch.py | 冻结实验身份、校验并汇总评测闭环 |
| watch_eval_to_obsidian.py | 读取逐轮记录、生成 Markdown、同步指定本地 vault |
| 本机 capture_video / build_gallery | 双视角视频和 GIF；未作为通用后端发布到本分支 |

## 输入约定

Linux、Python 3 标准库；批次内包含 `orchestrator.pid`、可选 `queue_status.json`，以及 `<scene>/plan.json`、`episodes.jsonl`、可选 `status.json`。plan 必须声明 `models`、`episodes_per_configuration`，支持 `modes`（默认 step_gt）。终态行含 model、mode、episode、outcome、success；失败阶段粗分类还用 contacts/max_lift。只适配此 schema，不是任意日志解析器。

默认每60秒更新，按场景/model/mode/episode去重，损坏行明确提示。正常完成和策略拒绝进入有效分母；其他终态单列无效。未结束回合不自动算失败。失败现象分类不等于因果判断。

## 运行

在已获授权的测试启动后，将正确的调度器 PID 写入 `orchestrator.pid`，并以独立进程运行：

```bash
python3 tools/a2d_eval_loop/watch_eval_to_obsidian.py \
  --batch /path/to/run \
  --vault /path/to/obsidian/A2D_tests \
  --interval 60
```

后台运行由启动器管理，输出重定向到批次 `auto_report.log`，记录 observer PID。独立于 evaluator 的进程能在 evaluator/调度器异常退出后收尾；不能保证整机断电后的即时同步。恢复后以同一命令加 `--once` 补档，再重新挂载 observer。PID 启动标识用于降低监控期间 PID 重用风险；首次挂载仍须操作者核对 PID。

## 输出和验收

- 批次：`AUTO_REPORT.md`、`auto_report_status.json`（源日志哈希、计数、警告）。
- Vault：`自动报告_<批次名>.md`、`00_自动测试报告索引.md`。
- 必须核对首份本地/vault报告存在、内容一致、observer存活，不能只把规则写进skill。
- 批次名须唯一；不同机器同名批次指向同一vault会覆盖同名自动报告。
- 仅原子替换本工具生成的文件；不重写人工历史报告。
- completed覆盖检查不等于科学验收；故障无效、初态未配对等限制仍需看原始契约。
- 单个模式是当前表格的主要使用范围；同一模型多个模式会合计显示，应分批运行或扩展表格分组。
- 文件系统同步失败写日志并重试；用户应保留日志和恢复补档流程。

GIF、视频上传和下一轮训练仍需各自人工确认，报告同步不授予这些权限。

## 报告开头

先显示“本轮目标与参数”简表：目标数量、模型、场景、执行方式、H/动作上限、种子、判据、视频和初态约束。参数从 plan.json 读取，缺失不猜测；结果表随后展示。改目标时创建新批次，不能修改历史报告代替修改运行计划。

## 唯一实现与 skill 分工

本仓库维护自动报告的唯一实现和测试。`a2d-grasp-evaluation` skill 只记录操作/验收流程；其旧脚本入口通过 `A2D_EVAL_REPO` 或本机 `config.json` 的 `eval_repo` 转发到此文件，不保留第二份报告代码。更换机器时重新配置 checkout 路径。
