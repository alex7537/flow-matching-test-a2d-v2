# A2D 评测闭环 v0

入口：[代码与运行说明](../tools/a2d_eval_loop/README.md)。

模型导出 → bundle 验收 → 既有 gRPC 评测 → 证据快照和汇总 → 下一轮任务草案 → 人工确认。

这是 L1 诊断控制层，不启动仿真或训练；每一项新增行动执行前先说明范围并取得人工确认。当前已授权任务可继续。

本实现补充 [原始 loop 方案](LOOP_FIRST_INSTANCE.md)，不把原 lifecycle 中 pending 的阶段标记成已通过。既有 gRPC runner 暂仍在部署端，尚未作为本工具的可移植执行后端发布。

验证：9 项标准库单元测试通过；本机历史 600 次尝试的六组成功/有效计数与原报告一致。新的 SSH 传输、训练提交、自动模型晋级未完成端到端验证。历史原始记录不包含在仓库中，单元测试可独立运行。

测试命令：

```bash
cd tools/a2d_eval_loop
python3 -m unittest discover -s tests -v
```
