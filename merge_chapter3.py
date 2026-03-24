# -*- coding: utf-8 -*-

# 读取所有部分
with open('IEEE-Transactions-LaTeX2e-templates-and-instructions/prefix.tex', 'r', encoding='utf-8') as f:
    prefix = f.read()

with open('chapter3_part1.txt', 'r', encoding='utf-8') as f:
    part1 = f.read()

with open('chapter3_part2.txt', 'r', encoding='utf-8') as f:
    part2 = f.read()

with open('chapter3_part3.txt', 'r', encoding='utf-8') as f:
    part3 = f.read()

with open('chapter3_part4.txt', 'r', encoding='utf-8') as f:
    part4 = f.read()

with open('IEEE-Transactions-LaTeX2e-templates-and-instructions/suffix.tex', 'r', encoding='utf-8') as f:
    suffix = f.read()

# 合并所有部分
chapter3 = part1 + part2 + part3 + part4

# 写入最终文件
output_path = 'IEEE-Transactions-LaTeX2e-templates-and-instructions/Casual-Qwen_Paper_CN.tex'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(prefix)
    f.write(chapter3)
    f.write(suffix)

print(f"Final file written to {output_path}")
print(f"Chapter 3 length: {len(chapter3)} characters")
