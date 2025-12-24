#!/usr/bin/env python3
"""
修复FAISS向量数据库加载问题
"""

import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings

def fix_vector_db():
    # 尝试两种不同的加载方式
    try:
        # 方式1：新版本参数
        embeddings = FakeEmbeddings(size=384)
        vector_store = FAISS.load_local(
            "./vector_db", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ 使用新版本参数加载成功")
    except TypeError as e:
        if "allow_dangerous_deserialization" in str(e):
            try:
                # 方式2：旧版本参数
                vector_store = FAISS.load_local(
                    "./vector_db", 
                    embeddings
                )
                print("✅ 使用旧版本参数加载成功")
            except Exception as e2:
                print(f"❌ 旧版本也失败: {e2}")
        else:
            print(f"❌ 其他错误: {e}")

if __name__ == "__main__":
    fix_vector_db()