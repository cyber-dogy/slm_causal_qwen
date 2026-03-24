# -*- coding: utf-8 -*-

# 第三章内容 - 分多个部分写入以避免JSON问题

# Part 1: 3.1和3.2节
part1 = r'''\section{未见工况任务形式化}

本节对未见工况下的 SLM 多模态缺陷识别任务进行形式化定义，为后续方法阐述奠定基础。

\subsection{问题定义}

设训练集 $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i, c_i)\}_{i=1}^{N_{\text{train}}}$ 和测试集 $\mathcal{D}_{\text{test}} = \{(\mathbf{x}_j, y_j, c_j)\}_{j=1}^{N_{\text{test}}}$，其中每个样本 $i$ 包含以下要素：
\begin{itemize}
    \item 输入模态集合 $\mathcal{M}_i \subseteq \{\text{rgb1}, \text{rgb2}, \text{ir}\}$，表示该样本包含的模态类型；
    \item 工况标识 $c_i \in \mathcal{C}$，其中 $\mathcal{C}$ 为所有工况的集合；
    \item 缺陷标签 $y_i \in \mathcal{Y} = \{\text{normal}, \text{HEW}, \text{LEL}\}$，分别为正常、匙孔缺陷（Higher Energy Weld）和缺乏熔合缺陷（Lack of Fusion）。
\end{itemize}

设每个模态 $m \in \mathcal{M}_i$ 对应的原始输入为 $\mathbf{x}_i^{(m)} \in \mathbb{R}^{H \times W \times C}$。模型的目标是学习映射函数 $f: \{(\mathbf{x}_i^{(m)})_{m \in \mathcal{M}_i}\} \rightarrow \mathcal{Y}$，使得在训练分布上最小化风险的同时，在未见工况的测试分布上保持良好泛化性能。

\textbf{训练与测试分布偏移}：与传统随机划分不同，本文采用条件级划分策略，要求训练集和测试集的工况集合互不相交，即
\begin{equation}
    \{c_i \mid (\mathbf{x}_i, y_i, c_i) \in \mathcal{D}_{\text{train}}\} \cap \{c_j \mid (\mathbf{x}_j, y_j, c_j) \in \mathcal{D}_{\text{test}}\} = \varnothing.
\end{equation}
这种划分强制模型学习跨工况稳定的缺陷表征，而非依赖特定工况的纹理、亮度或热背景等伪相关特征。

\textbf{核心问题}：在训练样本有限且存在显著工况分布偏移的条件下，如何利用大规模预训练视觉模型（如 Qwen2-VL）的先验知识，学习对缺陷本体敏感、对工况变化鲁棒的判别表示，实现未见工况下的稳定缺陷识别。

\subsection{条件级数据划分}

本文依托的实验仓库提供了两个数据域：

\textbf{主协议（data1 official\_cv）}：以 \texttt{condition\_uid} 为划分单元进行交叉验证，从源头上避免同一工况下样本的信息泄漏。该协议包含289个样本、22个不同工况，涵盖三种缺陷类别。这是本文方法设计和主要结论的基础评估协议。

\textbf{外部迁移压力测试（data2 transfer\_test）}：该数据集不参与模型选择和超参数调优，仅作为最终的外部迁移评估。\texttt{data2} 与 \texttt{data1} 在模型来源、工艺参数、纹理统计等方面存在显著差异，构成更接近真实工业部署场景的分布偏移压力测试。

表~\ref{tab:data\_stats\_cn} 给出了两个数据域的详细统计信息。

\begin{table}[!t]
\centering
\caption{主协议与迁移协议的数据统计}
\label{tab:data\_stats\_cn}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
数据域 & 样本数 & 工况数 & Normal & HEW & LEL \\
\hline
\texttt{data1} & 289 & 22 & 104 & 86 & 99 \\
\hline
\texttt{data2} & 404 & 22 & 167 & 128 & 109 \\
\hline
\end{tabular}
\end{table}

'''

with open('chapter3_part1.txt', 'w', encoding='utf-8') as f:
    f.write(part1)
    
print("Part 1 written to chapter3_part1.txt")
