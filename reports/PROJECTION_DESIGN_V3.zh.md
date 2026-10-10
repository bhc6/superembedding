# 拼接后为什么不一定需要 output projection

本项目每个固定属性方向产生一个 `u_k`，并非把普通 Transformer attention heads 直接取出拼接。各方向共享同一个 embedding projection 的参数；方向差异来自图像与选项文本经过独立 OmniJev 分支得到的隐藏状态。

原决策门控可直接作为投影输入：

\[
h_k=\mathrm{opt}(u_k)\odot\tanh(\mathrm{que}(z_q)),\qquad
s_k=\mathrm{normalize}\bigl(P(\mathrm{LayerNorm}(h_k))\bigr).
\]

`opt` 和 `que` 来自官方决策 head，`P` 输出 256 维。实验区分冻结门控和训练门控。raw 对照把 `h_k` 换成 `u_k`；residual 在归一化前增加独立的 raw 投影路径。这些方案都需要检索监督，不能假定原本对决策有用的几何空间天然适合余弦检索。

每个 slot 已单位归一化时，无需最终投影就可以得到固定长度、可索引的全局向量：

\[
e=\frac{1}{\sqrt K}[s_1;\ldots;s_K],\qquad
e(x)^\top e(y)=\frac1K\sum_k s_k(x)^\top s_k(y).
\]

这恰好是各属性余弦相似度的等权平均。按属性子集检索时，选出相应 slots 后重新归一化拼接即可，不必重新编码图片。

增加输出投影则是学习一个新的、通常会混合属性的距离度量：

\[
e_W=\frac{W[s_1;\ldots;s_K]}{\|W[s_1;\ldots;s_K]\|_2}.
\]

其相似度由 `WᵀW` 控制。等维投影用于检验这种度量适配是否有价值；降维投影还检验容量压缩的代价。若 `W` 为正交矩阵且保持维度，余弦检索结果不会改变，所以“多一层”本身不是改善的理由。

v3 的 joint concat、joint output equal、joint output small 使用相同的属性与全交集损失、相同采样方式、相同预算和相同 slot 初始化，只改变最终投影。raw 与 joint concat 网络相同、损失不同，因此 raw 对 joint output 的差值不能单独归因于 output projection。

保存压缩后的全局向量，并不等于仍能单独读取每种属性。如果应用还要按任意属性子集检索，必须另外保存 slots，或者另行研究能够从压缩向量解码/寻址属性的模型。v3 不把这种未实现的能力计入存储收益。

实测结论见 `RESULTS_V3.md` 与 `CONCLUSIONS_V3.zh.md`；理论上的可行性、相同训练预算下的结果和模型最优可达性能是不同层次的结论。
