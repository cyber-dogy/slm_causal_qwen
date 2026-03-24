# -*- coding: utf-8 -*-

with open('Casual-Qwen_Paper_CN.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 更新引言部分的引用
# 原: \cite{ref_pbf_review,ref_optical_defect} → 新: \cite{ref_defect_formation_slm}
content = content.replace('\\cite{ref_pbf_review,ref_optical_defect}', '\\cite{ref_defect_formation_slm}')

# 原: \cite{ref_ir_defect} → 新: \cite{ref_in_situ_monitoring_review}
content = content.replace('\\cite{ref_ir_defect}', '\\cite{ref_in_situ_monitoring_review}')

# 原: \cite{ref_multisensor_am,ref_multisensor_platform} → 新: \cite{ref_lpbf_sensing_review,ref_intelligent_defect_pbf}
content = content.replace('\\cite{ref_multisensor_am,ref_multisensor_platform}', '\\cite{ref_lpbf_sensing_review,ref_intelligent_defect_pbf}')

# 原: \cite{ref_irm,ref_groupdro} → 删除（用户未提供相关文献）
content = content.replace(' \\cite{ref_irm,ref_groupdro}', '')

# 删除临时文件
with open('Casual-Qwen_Paper_CN.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Citations in introduction updated!')
