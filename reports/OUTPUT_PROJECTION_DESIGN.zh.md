# 拼接后是否需要 output projection

不是必要层。设每属性单位向量为 sₖ，直接记忆向量是 normalize(concat(s₁,…,sK))。固定 K 时，两图点积恰好为所有属性 cosine 的平均。指定一个属性就取相应 slot；指定两个属性就拼接两个 slot。这种分数可解释且无需重新编码对象。

增加 W 后得到 normalize(W concat(s₁,…,sK))，相似度受 WᵀW 影响，允许属性间交叉项。它能压缩、重新加权及学习组合检索，但压缩时会丢失信息，而且不能直接把投影结果切块解释为原属性。

因此模型同时返回 slots 和 global embedding。实验比较：

| 结构 | slot loss | joint loss | 全局维度 |
|---|---|---|---|
| raw | 是 | 否 | 768 |
| joint_concat | 是 | 是 | 768 |
| joint_out768 | 是 | 是 | 768 |
| joint_out256 | 是 | 是 | 256 |

后三种的采样预算与目标一致；是否加 projection 由三属性交集检索、属性检索保持程度、存储成本共同判断。若保存 slots 又保存 global，实际存储是 768+global_dim，不能仅声称压缩至256。若只存256d投影向量，则失去当前实现按 slot 自由选属性的能力。

门控残差架构：normalize(Pg(LN(opt(u)⊙tanh(que(zq)))) + Pu(LN(u)))。残差保留决策门控可能抑制的信息；参数更多，不能把任何提升全归因于决策证据迁移。
