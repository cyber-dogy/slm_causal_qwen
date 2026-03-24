import codecs

# Read the main file
with codecs.open('IEEE-Transactions-LaTeX2e-templates-and-instructions/Casual-Qwen_Paper_CN.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Read new section content
with codecs.open('new_section_33.tex', 'r', encoding='utf-8') as f:
    new_content = f.read()

# Find line numbers - merge sections 3.3 and 3.4
start_line = None
end_line = None
for i, line in enumerate(lines):
    if '大模型轻量适配与真微调对照' in line and '\\subsection' in line:
        start_line = i
    if start_line and '实验与分析' in line and '\\section' in line:
        end_line = i
        break

print(f'Start: {start_line}, End: {end_line}')

if start_line is not None and end_line is not None:
    # Replace lines
    new_lines = lines[:start_line] + [new_content] + lines[end_line:]
    
    with codecs.open('IEEE-Transactions-LaTeX2e-templates-and-instructions/Casual-Qwen_Paper_CN.tex', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('Section 3.3 replaced successfully')
else:
    print('Could not find section boundaries')
