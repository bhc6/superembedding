# V5 导出与正式评测的一致性修复

> 2026-10-12 协议更新：下文的 validation-best/finalizer 描述属于旧流程。当前 Step556 延续训练将旧验证仅用作诊断，正式候选固定为 terminal Step2000，须完成独立测试集审查及同一 checkpoint 的完整 LoRA+head 导出重载核验后才可评测/发布。旧 `best.pt` 不具备正式选模资格；见 `STEP556_CONTINUATION.zh.md`。完整张量导出与重载一致性的技术要求仍然适用。

2026-10-11 核对确认：原 `train_general.py` 在训练结束时直接保存最后一步 `embedding_head.pt`，正式评测原先读取验证选中的 `best.pt`。二者可能步号不同，且 head-only 不含训练后的 LoRA；旧 `encode_image.py` 的输入构造还是属性图片编码器，不是本阶段的 `GeneralOmniEncoder`。因此把该 head 当成正式评测模型发布或推理，确实不正确。

核对时 V5 正在训练，尚未生成最终 head 或正式 V5 模型测试结果，也未发布。不能据此推定 V3/V4 所有历史结果错误：旧实验有独立导出验证，V3 的 432 个 head 验证重放最大指标差为 0；V4 的缓存 head 验证也记录了重放结果。

## 修复后的唯一正式产物

- `best.pt`：验证选中的完整训练恢复状态，含 LoRA、projection、optimizer、RNG。只用验证选择，不用测试选择。
- `export/general_model.pt`：直接从同一 `best.pt` 导出所有 304 个训练张量（300 个 LoRA、4 个投影张量）、源 checkpoint SHA256、步号、基础模型身份、方向顺序及通用输入构造约定。它依赖固定原始 OmniJev adapter 和 Qwen backbone，不含 optimizer，不冒充自足的完整 backbone 文件。
- `export/general_model.audit.json`：逐张量一致性及文本/图片/图文真实输入重载误差。正式评测必须等这项通过。
- 原 `embedding_head.pt`：训练结束后由 finalizer 保留为 `last_step_head_only_DO_NOT_DEPLOY.pt`，不作为发布推理模型。当前 fingerprinted 训练代码不热改，因此 finalizer 明确处理此旧产物。

`finish_general_stage.py` 检查训练完成、best 步号与状态一致，导出并验证模型包，然后将同一包路径交给 ScienceQA 与通用迁移评测。正式评测与新推理入口共用 `load_bundle_encoder()`，保持 `GeneralOmniEncoder`、4 个方向、256 tokens、checkpoint 像素预算和投影配置一致。旧 `encode_image.py` 会拒绝该通用 basis 的独立 head，提示改用正确入口。

```powershell
. ./scripts/use_d_storage.ps1
.venv/Scripts/python.exe scripts/encode_general.py --model D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/export/general_model.pt --image path/to/image.jpg --text "需要编码的对象描述" --output runs/general_embedding.pt
```

上面的正式文件要到验证选模和训练完成后才会生成。当前工程核验使用第 215 步的独立只读快照，**它不是选中的最终模型**：在 CPU 上分别从 checkpoint 和导出包全新加载，304 个训练张量完全相等；文本、图片、图文三种真实输入的 embedding 最大绝对差均为 0。证据：`artifacts/results/v5/general_export_consistency_probe.json`。最后选中的模型会再次在 CUDA 正式评测前运行同一检查，不能用此次快照核验替代。

新增测试覆盖 checkpoint 步号/权重保留、禁止不同 checkpoint 覆盖同一产物、缺 LoRA 拒绝导出、head-only 拒绝加载、输入构造约定变更拒绝加载。模型主结果仍待训练和正式冻结评测完成，全部研究完成后才提交 Git。
