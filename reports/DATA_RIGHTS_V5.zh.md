# V5 数据来源与发布范围核对（未完成）

核对日期：2026-10-11。以下载时的一手页面、许可文件及 SHA256 为依据。以下区分数据、标注、项目代码；网页中 JavaScript 的 Apache/MIT 标头不当作数据许可。

the_cauldron 固定 revision 的数据卡明确说每个子数据集受各自许可约束，聚合方拥有权利的 prompts 为 CC-BY-4.0；不存在可以覆盖全部图片的一揽子聚合许可。

| 来源 | 一手证据 | 当前核对结论 |
|---|---|---|
| Localized Narratives | [项目站](https://google.github.io/localized-narratives/) | 标注 CC BY 4.0；页面另列每张图的 image license，不能推定所有图像许可相同 |
| TextCaps | [项目站](https://textvqa.org/textcaps/) | 已取得页面，但未提取到明确数据条款；图片/标注许可仍需核对 |
| Screen2Words | [官方 README](https://github.com/google-research-datasets/screen2words) | README 指向 RICO 的 22417 个界面，未给出数据许可；尝试的 LICENSE 路径返回 404，不能据此断言无许可 |
| Chart2Text | [官方仓库](https://github.com/vis-nlp/Chart-to-text) | 仓库 LICENSE 为 GPL-3.0；原始图表内容及标注的再分发范围仍需单独确认 |
| ChartQA | [官方仓库](https://github.com/vis-nlp/ChartQA) | 仓库 LICENSE 为 GPL-3.0；不自动视为全部原始第三方图表的许可 |
| DocVQA | [数据页面](https://www.docvqa.org/datasets/docvqa) | 页面的脚本许可不能解释为文档图像许可，数据条款尚未完成核对 |
| CLEVR | [官方项目站](https://cs.stanford.edu/people/jcjohns/clevr/) | 页面明确写 All data is released under CC BY 4.0 |
| VSR | [官方仓库](https://github.com/cambridgeltl/visual-spatial-reasoning) | 项目声明 Apache-2.0；原图权利不由这句项目声明自动覆盖 |
| AI2D | 原始来源留出 | 当前请求的 README 路径返回 404，数据许可未核实；仅用于留出评测，不进入训练 |
| ScienceQA | [官方 README 的 Licenses](https://github.com/lupantech/ScienceQA#warning-licenses) | 数据明确 CC BY-NC-SA 4.0；代码一行文字称 MIT 但链接指向 CC 页面，不混用。仅来源留出及官方评测，不进入训练 |

模型上游：本地固定版本 Qwen3.5-0.8B 与 OmniJev 的许可文件保留。代码、数据和衍生权重的发布声明应分别核验；不能因 backbone 为 Apache-2.0 就宣布所有训练数据无限制。此处不推定具体数据条款对衍生权重的法律效果。

预备发布的真实数据清单只含来源、revision、原始行号、本地 split/order 和图像哈希，不含原图、问答、描述或本机绝对路径。用户可按上游条款获取原始数据，再核对身份。原创 `evidence_diagnostic_v1` 的图片和标注单独标记 CC0，不把该许可扩展到其他数据。

当前仍需完成 TextCaps/RICO/DocVQA/AI2D 及第三方图表/图片的来源核对，以及衍生权重模型卡中的准确使用说明。此报告不是全部发布事项已完成的证明。代码和训练继续保留，按用户要求待全部完成后一次性提交。

一手抓取证据及哈希：`artifacts/results/v5/data_rights_audit.json`。原始网页本地保留在忽略目录 `runs/v5/license_sources`，不作为仓库授权文件重新分发。
