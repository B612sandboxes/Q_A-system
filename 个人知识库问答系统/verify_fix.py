#!/usr/bin/env python3
"""
修复验证脚本
验证pwd修复和基础功能
"""

import sys
import os

def verify_pwd_fix():
    """验证pwd修复"""
    print("1️⃣ 验证pwd模块修复...")
    try:
        # 尝试导入修复
        from pwd_fix import fix_pwd_module
        if fix_pwd_module():
            # 验证pwd模块可用
            import pwd
            print(f"   pwd模块类型: {type(pwd)}")
            print(f"   pwd方法: {dir(pwd)[:5]}...")
            print("✅ pwd修复验证成功")
            return True
    except Exception as e:
        print(f"❌ pwd修复验证失败: {e}")
        return False

def verify_imports():
    """验证关键导入"""
    print("\n2️⃣ 验证关键模块导入...")
    
    modules = [
        ("langchain.document_loaders", "PyPDFLoader"),
        ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),
        ("langchain.embeddings", "HuggingFaceEmbeddings"),
        ("langchain.vectorstores", "FAISS"),
        ("langchain.chains", "RetrievalQA"),
    ]
    
    all_ok = True
    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            print(f"   ✅ {module_path}.{class_name} 导入成功")
        except ImportError as e:
            print(f"   ❌ {module_path}.{class_name} 导入失败: {e}")
            all_ok = False
    
    return all_ok

def verify_directories():
    """验证目录结构"""
    print("\n3️⃣ 验证目录结构...")
    
    required_dirs = [
        "./knowledge_base",
        "./vector_db",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ 目录存在: {dir_path}")
        else:
            print(f"   ⚠️  目录不存在: {dir_path}")
            print(f"      正在创建...")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"      ✅ 创建成功")
            except Exception as e:
                print(f"      ❌ 创建失败: {e}")
                all_ok = False
    
    return all_ok

def verify_sample_document():
    """验证示例文档"""
    print("\n4️⃣ 验证知识库文档...")
    
    knowledge_base = "./knowledge_base"
    
    if not os.path.exists(knowledge_base):
        print("   ⚠️  知识库目录不存在")
        return False
    
    files = os.listdir(knowledge_base)
    supported_ext = ['.pdf', '.txt', '.docx', '.md']
    
    supported_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_ext]
    
    if supported_files:
        print(f"   ✅ 找到{len(supported_files)}个支持的文件:")
        for f in supported_files[:3]:  # 显示前3个
            print(f"      - {f}")
        if len(supported_files) > 3:
            print(f"      等{len(supported_files)}个文件")
        return True
    else:
        print("   ⚠️  知识库中没有支持的文件格式")
        print("      支持格式: .pdf, .txt, .docx, .md")
        print("      请将文档放入knowledge_base文件夹")
        return False

def main():
    """主验证函数"""
    print("=" * 60)
    print("🔍 个人知识库问答系统 - 环境验证")
    print("=" * 60)
    
    # 添加当前目录到路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    results = []
    
    # 运行验证
    results.append(("pwd修复", verify_pwd_fix()))
    results.append(("模块导入", verify_imports()))
    results.append(("目录结构", verify_directories()))
    results.append(("知识库文档", verify_sample_document()))
    
    print("\n" + "=" * 60)
    print("📋 验证结果汇总:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "⚠️  警告"
        print(f"{test_name:15} {status}")
    
    print(f"\n📊 通过率: {passed}/{total}")
    
    if passed >= 3:  # 允许知识库文档不存在
        print("\n🎉 验证通过！可以启动系统了。")
        print("   运行命令: python app.py")
        return True
    else:
        print("\n⚠️  验证未通过，请检查安装和配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)