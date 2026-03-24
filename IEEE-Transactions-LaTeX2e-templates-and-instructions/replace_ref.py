# -*- coding: utf-8 -*-

# 读取新的参考文献
with open('ref_new.bib', 'r', encoding='utf-8') as f:
    new_refs = f.read()

# 读取原文件
with open('Casual-Qwen_Paper_CN.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到参考文献部分并替换
start_marker = '\\begin{thebibliography}'
end_marker = '\\end{thebibliography}'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker) + len(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + new_refs + content[end_idx:]
    with open('Casual-Qwen_Paper_CN.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('References replaced successfully!')
else:
    print(f'Could not find markers: start={start_idx}, end={end_idx}')

# 删除临时文件
import os
os.remove('ref_new.bib')
