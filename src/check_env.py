"""
Qwen2-VL SLM 训练环境检查脚本

该脚本检查环境是否正确配置以进行训练。
在开始训练前运行此脚本，以识别缺失的依赖项或配置问题。
"""

import sys
import os


def check_python_version():
    """检查 Python 版本。"""
    print("=" * 60)
    print("检查 Python 版本")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠️  警告: 推荐使用 Python 3.9+")
        return False
    else:
        print("✅ Python 版本正常")
        return True


def check_package(package_name, import_name=None, min_version=None):
    """检查包是否已安装。"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        print(f"  {package_name}: {version}", end="")
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f" ⚠️  (推荐 >= {min_version})")
                return False
        
        print(" ✅")
        return True
    except ImportError:
        print(f"  {package_name}: 未安装 ❌")
        return False


def check_pytorch():
    """检查 PyTorch 安装。"""
    print("\n" + "=" * 60)
    print("检查 PyTorch")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 检查 CUDA
        if torch.cuda.is_available():
            print(f"CUDA 可用: ✅")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    总显存: {props.total_memory / 1e9:.2f} GB")
        else:
            print("CUDA 可用: ❌ (仅 CPU)")
            print("⚠️  警告: CPU 训练会非常慢")
        
        # 检查 bf16 支持
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("bfloat16 支持: ✅")
        else:
            print("bfloat16 支持: ❌ (将使用 float16 或 float32)")
        
        return True
    except ImportError:
        print("PyTorch: 未安装 ❌")
        print("安装命令: pip install torch torchvision")
        return False


def check_transformers():
    """检查 Transformers 安装。"""
    print("\n" + "=" * 60)
    print("检查 Transformers")
    print("=" * 60)
    
    try:
        import transformers
        print(f"Transformers 版本: {transformers.__version__}")
        
        # 检查 Qwen2VL 是否可用
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            print("Qwen2-VL 模型: ✅ 可用")
        except ImportError:
            print("Qwen2-VL 模型: ❌ 不可用")
            print("   更新命令: pip install transformers -U")
            return False
        
        return True
    except ImportError:
        print("Transformers: 未安装 ❌")
        print("安装命令: pip install transformers")
        return False


def check_peft():
    """检查 PEFT 安装。"""
    print("\n" + "=" * 60)
    print("检查 PEFT (用于 LoRA)")
    print("=" * 60)
    
    try:
        import peft
        print(f"PEFT 版本: {peft.__version__}")
        
        # 检查 LoRA 可用性
        try:
            from peft import LoraConfig, get_peft_model
            print("LoRA 支持: ✅ 可用")
            return True
        except ImportError:
            print("LoRA 支持: ❌ 不可用")
            return False
    except ImportError:
        print("PEFT: 未安装 ❌")
        print("安装命令: pip install peft")
        return False


def check_other_packages():
    """检查其他必需的包。"""
    print("\n" + "=" * 60)
    print("检查其他依赖")
    print("=" * 60)
    
    packages = [
        ('numpy', 'numpy', '1.24.0'),
        ('pandas', 'pandas', '2.0.0'),
        ('PIL', 'PIL', None),
        ('scikit-learn', 'sklearn', '1.4.0'),
        ('matplotlib', 'matplotlib', '3.8.0'),
        ('seaborn', 'seaborn', '0.13.0'),
        ('tqdm', 'tqdm', None),
    ]
    
    all_ok = True
    for pkg, import_name, min_ver in packages:
        if not check_package(pkg, import_name, min_ver):
            all_ok = False
    
    return all_ok


def check_optional_packages():
    """检查可选包。"""
    print("\n" + "=" * 60)
    print("检查可选依赖")
    print("=" * 60)
    
    optional = [
        ('bitsandbytes', 'bitsandbytes', '0.43.0'),
        ('accelerate', 'accelerate', '0.30.0'),
        ('tensorboard', 'tensorboard', '2.15.0'),
        ('flash_attn', 'flash_attn', None),
    ]
    
    for pkg, import_name, min_ver in optional:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  {pkg}: {version} ✅")
        except ImportError:
            print(f"  {pkg}: 未安装 (可选)")


def check_data_paths():
    """检查数据路径是否存在。"""
    print("\n" + "=" * 60)
    print("检查数据路径")
    print("=" * 60)
    
    # 当前主流程默认路径
    default_paths = [
        "/home/gjw/code/SLM_data/processed_qwen_balanced/train_manifest.jsonl",
        "/home/gjw/code/SLM_data/processed_qwen_balanced/val_manifest.jsonl",
        "/home/gjw/code/SLM_data/processed_qwen_balanced/test_manifest.jsonl",
        "/home/gjw/code/SLM_data/processed_qwen_data1_protocol/quick_dev/train_manifest.jsonl",
        "/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_01/train_manifest.jsonl",
    ]
    
    all_exist = True
    for path in default_paths:
        if os.path.exists(path):
            print(f"  {path}: ✅ 存在")
        else:
            print(f"  {path}: ❌ 未找到")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  部分数据文件未找到。请检查你的数据路径。")
    
    return all_exist


def estimate_memory():
    """估算显存需求。"""
    print("\n" + "=" * 60)
    print("显存需求估算")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"可用 GPU 显存: {gpu_memory:.2f} GB")
            
            if gpu_memory < 16:
                print("针对你的 GPU 推荐设置:")
                print("  --finetune_mode qlora")
                print("  --load_in_4bit true")
                print("  --per_device_train_batch_size 1")
                print("  --gradient_accumulation_steps 8")
            elif gpu_memory < 32:
                print("针对你的 GPU 推荐设置:")
                print("  --finetune_mode qlora")
                print("  --load_in_4bit true")
                print("  --per_device_train_batch_size 2")
                print("  --gradient_accumulation_steps 4")
            else:
                print("针对你的 GPU 推荐设置:")
                print("  --finetune_mode lora")
                print("  --load_in_4bit false")
                print("  --per_device_train_batch_size 4")
                print("  --gradient_accumulation_steps 2")
        else:
            print("未检测到 GPU。训练将使用 CPU (非常慢)。")
            print("考虑使用带 GPU 的机器进行训练。")
    except:
        print("无法估算显存需求。")


def main():
    """运行所有检查。"""
    print("\n" + "=" * 60)
    print("Qwen2-VL SLM 训练 - 环境检查")
    print("=" * 60)
    
    checks = [
        ("Python 版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("PEFT", check_peft),
        ("其他包", check_other_packages),
        ("可选包", check_optional_packages),
        ("数据路径", check_data_paths),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ 检查 {name} 时出错: {e}")
            results[name] = False
    
    # 显存估算
    try:
        estimate_memory()
    except Exception as e:
        print(f"无法估算显存: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    critical = ["Python 版本", "PyTorch", "Transformers"]
    all_critical_ok = all(results.get(c, False) for c in critical)
    
    if all_critical_ok:
        print("✅ 环境已准备好进行训练!")
        print("\n下一步:")
        print("  1. 更新 run_train_eval.ps1 中的数据路径")
        print("  2. 运行: .\\run_train_eval.ps1 -RunName 'my_first_run'")
    else:
        print("❌ 环境存在问题需要解决。")
        print("\n请安装缺失的依赖:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
