# -*- coding: utf-8 -*-

# Part 2: 3.3节提出方法和4.1多模态特征提取器
part2 = r'''\section{提出方法}

针对上述未见工况任务，本文提出一种基于大模型先验与因果一致性约束的多模态缺陷识别框架。如图~\ref{fig:framework\_cn}所示，该方法包含三个核心组件：多模态特征提取器、因果一致性表征融合模块、以及大模型先验的轻量适配机制。

\begin{figure*}[!t]
\centering
% TODO: 替换为正式方法框架图。
\fbox{\rule{0pt}{1.9in}\rule{0.95\textwidth}{0pt}}
\caption{本文方法框架：多模态特征提取—因果一致性约束—轻量判别适配的三级结构。}
\label{fig:framework\_cn}
\end{figure*}

\subsection{多模态特征提取器}

本文将 Qwen2-VL 的视觉编码器 $f_{\theta_v}$ 作为预训练视觉先验。对于样本 $i$ 的任意输入模态 $m \in \mathcal{M}_i$，视觉编码器首先将图像 $\mathbf{x}_i^{(m)}$ 编码为视觉 token 序列：
\begin{equation}
    \mathbf{T}_i^{(m)} = f_{\theta_v}\left(\mathbf{x}_i^{(m)}\right) \in \mathbb{R}^{P \times d},
    \label{eq:visual_token}
\end{equation}
其中 $P$ 为空间合并后的视觉 token 数量，$d$ 为视觉隐藏维度（本文使用 Qwen2-VL-2B-Instruct，$d=1536$）。

为获得紧凑的单模态表征，对各模态的视觉 token 序列执行平均池化：
\begin{equation}
    \mathbf{z}_i^{(m)} = \frac{1}{P} \sum_{p=1}^{P} \mathbf{T}_{i,p}^{(m)} \in \mathbb{R}^{d},
    \label{eq:avg_pool}
\end{equation}
其中 $\mathbf{T}_{i,p}^{(m)}$ 表示第 $p$ 个视觉 token。该池化操作保留了模态级的全局语义信息，同时显著降低了后续融合的维度复杂度。

'''

with open('chapter3_part2.txt', 'w', encoding='utf-8') as f:
    f.write(part2)
    
print("Part 2 written to chapter3_part2.txt")
