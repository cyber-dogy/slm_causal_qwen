# -*- coding: utf-8 -*-

# 读取原文件
with open('IEEE-Transactions-LaTeX2e-templates-and-instructions/Casual-Qwen_Paper_CN.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 提取前59行
prefix = ''.join(lines[:59])

# 提取第199行及以后（从0开始索引是198）
suffix = ''.join(lines[198:])

# 保存前缀和后缀
with open('IEEE-Transactions-LaTeX2e-templates-and-instructions/prefix.tex', 'w', encoding='utf-8') as f:
    f.write(prefix)

with open('IEEE-Transactions-LaTeX2e-templates-and-instructions/suffix.tex', 'w', encoding='utf-8') as f:
    f.write(suffix)

print('Prefix and suffix saved')
