# -*- coding: utf-8 -*-

with open('Casual-Qwen_Paper_CN.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 删除相关工作A中不存在的引用
# Abdelrahman 等 \cite{ref_optical_defect} → Abdelrahman 等（删除引用）
content = content.replace('Abdelrahman 等 \\cite{ref_optical_defect}', '相关研究')

# Bartlett 等 \cite{ref_in_situ_monitoring_review} → 保持（但这个引用重复使用了）
# 实际上Bartlett不在用户提供的文献中，需要删除
content = content.replace('Bartlett 等 \\cite{ref_in_situ_monitoring_review}', '相关研究')

# Chua 等 \cite{ref_pbf_review} → Wang 等 \cite{ref_intelligent_defect_pbf}
content = content.replace('Chua 等 \\cite{ref_pbf_review}', 'Wang 等 \\cite{ref_intelligent_defect_pbf}')

# Petrich 等 \cite{ref_multisensor_am} → 相关研究
content = content.replace('Petrich 等 \\cite{ref_multisensor_am}', '相关研究')

# Maucher 等 \cite{ref_multisensor_platform} → 相关研究  
content = content.replace('Maucher 等 \\cite{ref_multisensor_platform}', '相关研究')

# 删除相关工作B中的不存在引用
# Arjovsky 等 \cite{ref_irm} → 删除
content = content.replace('Arjovsky 等 \\cite{ref_irm} 指出，', '研究表明，')

# Sagawa 等 \cite{ref_groupdro} → 删除
content = content.replace('Sagawa 等 \\cite{ref_groupdro} 进一步表明，', '此外，')

# 删除临时文件
with open('Casual-Qwen_Paper_CN.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Citations in related work updated!')
