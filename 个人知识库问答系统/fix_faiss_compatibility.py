#!/usr/bin/env python3
"""
FAISS版本兼容性修复脚本
解决 allow_dangerous_deserialization 参数错误
"""

import os
import sys

def check_faiss_version():
    """检查FAISS版本和兼容性"""
    try:
        import faiss
        version = faiss.__version__
        print(f"🔍 当前FAISS版本: {version}")
        
        # 检查版本兼容性
        if version >= '1.10.0':
            print("✅ 检测到新版本FAISS，使用新API")
            return "new"
        else:
            print("✅ 检测到旧版本FAISS，使用兼容API")
            return "old"
            
    except ImportError as e:
        print(f"❌ FAISS导入失败: {e}")
        return "error"

def create_compatible_loader():
    """创建兼容的FAISS加载器"""
    version_type = check_faiss_version()
    
    if version_type == "new":
        # 新版本不需要 allow_dangerous_deserialization 参数
        loader_code = '''
def load_faiss_safe(embeddings, vector_db_path):
    """新版本FAISS加载器"""
    from langchain_community.vectorstores import FAISS
    try:
        # 新版本API
        vector_store = FAISS.load_local(vector_db_path, embeddings)
        return vector_store
    except Exception as e:
        print(f"新版本加载失败: {e}")
        return None
'''
    else:
        # 旧版本需要参数
        loader_code = '''
def load_faiss_safe(embeddings, vector_db_path):
    """旧版本FAISS加载器"""
    from langchain_community.vectorstores import FAISS
    try:
        # 旧版本API
        vector_store = FAISS.load_local(
            vector_db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"旧版本加载失败: {e}")
        return None
'''
    
    return loader_code

def main():
    """主函数"""
    print("=" * 50)
    print("🔧 FAISS版本兼容性修复工具")
    print("=" * 50)
    
    # 检查版本
    version_type = check_faiss_version()
    
    # 生成兼容代码
    loader_code = create_compatible_loader()
    
    # 保存修复代码
    with open("faiss_compat.py", "w", encoding="utf-8") as f:
        f.write('''"""
FAISS兼容性模块
自动处理版本差异
"""

''' + loader_code + '''

# 自动检测并加载
def auto_load_faiss(embeddings, vector_db_path):
    """自动检测版本并加载FAISS索引"""
    try:
        # 先尝试新版本API
        vector_store = FAISS.load_local(vector_db_path, embeddings)
        print("✅ 使用新版本API加载成功")
        return vector_store
    except TypeError as e:
        if "allow_dangerous_deserialization" in str(e):
            # 回退到旧版本API
            try:
                vector_store = FAISS.load_local(
                    vector_db_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("✅ 使用旧版本API加载成功")
                return vector_store
            except Exception as e2:
                print(f"❌ 旧版本API也失败: {e2}")
                return None
        else:
            print(f"❌ 其他错误: {e}")
            return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
''')
    
    print("✅ 兼容性修复代码已生成: faiss_compat.py")
    print("💡 使用方法: from faiss_compat import auto_load_faiss")

if __name__ == "__main__":
    main()