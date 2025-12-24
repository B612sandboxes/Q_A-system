#!/usr/bin/env python3
"""
环境测试脚本
检查系统环境和依赖
"""

import sys
import platform
import subprocess

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python版本: {sys.version}")
        return True
    else:
        print(f"❌ Python版本过低: {sys.version}")
        print("   需要Python 3.8+")
        return False

def check_os():
    """检查操作系统"""
    print(f"🔍 操作系统: {platform.system()} {platform.release()}")
    return True

def check_imports():
    """检查关键导入"""
    imports_to_check = [
        ("langchain", "langchain"),
        ("faiss", "faiss-cpu"),
        ("pypdf", "pypdf"),
    ]
    
    all_ok = True
    for module_name, package_name in imports_to_check:
        try:
            __import__(module_name)
            print(f"✅ {module_name} 导入成功")
        except ImportError:
            print(f"❌ {module_name} 导入失败")
            print(f"   请运行: pip install {package_name}")
            all_ok = False
    
    return all_ok

def check_pwd_fix():
    """检查pwd修复"""
    try:
        from pwd_fix import fix_pwd_module
        if fix_pwd_module():
            print("✅ pwd模块修复成功")
            return True
    except Exception as e:
        print(f"❌ pwd模块修复失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("🧪 环境测试开始")
    print("=" * 50)
    
    tests = [
        ("Python版本", check_python_version),
        ("操作系统", check_os),
        ("依赖导入", check_imports),
        ("pwd修复", check_pwd_fix),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n📈 通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以运行主程序了。")
        print("   运行命令: python app.py")
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装。")

if __name__ == "__main__":
    main()