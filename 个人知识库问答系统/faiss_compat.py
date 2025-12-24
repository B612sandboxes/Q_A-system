"""
FAISS兼容性模块
自动处理版本差异
"""


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
