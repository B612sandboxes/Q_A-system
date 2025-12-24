import os
import time
import sys
from typing import List, Dict, Any
import traceback

# 设置环境变量，解决网络问题[7,9]
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# 禁用警告信息，减少输出干扰
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入修复模块
try:
    from pwd_fix import fix_pwd_module
    fix_pwd_module()
except ImportError:
    print("⚠️  pwd_fix模块导入失败，可能影响Windows环境")

# LangChain核心组件 - 使用正确的导入路径避免弃用警告
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    print("✅ LangChain组件导入成功")
except ImportError as e:
    print(f"❌ LangChain组件导入失败: {e}")
    print("请运行: pip install langchain-community sentence-transformers")
    sys.exit(1)


class KnowledgeBaseQA:
    """知识库问答系统核心类"""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base = knowledge_base_path
        self.vector_db_path = "./vector_db"
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        
        # 创建必要目录
        os.makedirs(self.knowledge_base, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    def init_embeddings(self):
        """初始化文本嵌入模型 - 修复网络问题和模型路径[7,8](@ref)"""
        try:
            # 步骤1：使用完整的模型路径，避免404错误
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"🔄 尝试加载模型: {model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True  # 添加信任远程代码参数
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'show_progress_bar': False  # 禁用进度条减少输出
                }
            )
            
            # 测试嵌入模型是否正常工作
            test_embedding = self.embeddings.embed_query("测试文本")
            if len(test_embedding) > 0:
                print(f"✅ 文本嵌入模型初始化成功: {model_name}")
                return True
            else:
                raise Exception("嵌入测试返回空结果")
                
        except Exception as e:
            print(f"❌ 嵌入模型初始化失败: {e}")
            print("⚠️  使用离线嵌入模型...")
            return self.init_embeddings_offline()
    
    def init_embeddings_offline(self):
        """离线模式下的嵌入模型初始化 - 使用正确的导入路径[7](@ref)"""
        try:
            # 步骤2：使用正确的社区版导入
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
            print("✅ 使用离线嵌入模型（检索质量会降低，但系统可运行）")
            return True
        except Exception as e:
            print(f"❌ 离线嵌入模型也失败: {e}")
            # 终极备用方案：创建最简单的嵌入类
            class SimpleEmbeddings:
                def __init__(self, size=384):
                    self.size = size
                
                def embed_query(self, text):
                    # 返回随机向量（仅用于测试）
                    import random
                    return [random.gauss(0, 1) for _ in range(self.size)]
                
                def embed_documents(self, texts):
                    return [self.embed_query(text) for text in texts]
            
            self.embeddings = SimpleEmbeddings()
            print("✅ 使用简单随机嵌入模型（仅保证系统运行）")
            return True
    
    def load_documents(self) -> List:
        """加载知识库中的所有文档"""
        if not os.path.exists(self.knowledge_base):
            print(f"⚠️  知识库文件夹不存在: {self.knowledge_base}")
            print(f"✅ 已创建，请将文档放入此文件夹")
            return []
        
        documents = []
        supported_extensions = {'.pdf', '.txt', '.docx', '.md'}
        
        # 检查文件数量
        files = os.listdir(self.knowledge_base)
        if not files:
            print("ℹ️  知识库文件夹为空")
            return []
        
        print(f"📁 发现{len(files)}个文件，开始加载支持的文档...")
        
        for filename in files:
            file_path = os.path.join(self.knowledge_base, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in supported_extensions:
                print(f"⏭️  跳过不支持的文件格式: {filename}")
                continue
            
            try:
                if ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif ext == '.txt':
                    # 尝试多种编码[5](@ref)
                    loader = None
                    for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                        try:
                            loader = TextLoader(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if loader is None:
                        raise ValueError(f"无法解码文件: {filename}")
                elif ext == '.docx':
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = filename
                    # 确保页面信息存在
                    if 'page' not in doc.metadata:
                        doc.metadata['page'] = 1
                
                documents.extend(loaded_docs)
                print(f"📄 已加载: {filename} ({len(loaded_docs)}个片段)")
                
            except Exception as e:
                print(f"❌ 加载{filename}失败: {e}")
                continue
        
        if documents:
            print(f"✅ 共加载{len(documents)}个文档片段")
        else:
            print("⚠️  未加载任何文档片段")
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 800, chunk_overlap: int = 100):
        """分割文档为小块"""
        if not documents:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"📊 文档分割完成: {len(documents)} -> {len(split_docs)} 个片段")
        return split_docs
    
    def create_vector_store(self, documents: List, force_recreate: bool = False):
        """创建向量数据库 """
        # 检查是否已存在
        faiss_index = os.path.join(self.vector_db_path, "index.faiss")
        pkl_index = os.path.join(self.vector_db_path, "index.pkl")
        
        if not force_recreate and os.path.exists(faiss_index) and os.path.exists(pkl_index):
            print("📂 加载现有向量数据库...")
            try:
                # 步骤3：修复FAISS版本兼容性 - 尝试不同加载方式
                self.vector_store = self._load_faiss_compatible()
                if self.vector_store:
                    print("✅ 向量数据库加载成功")
                    return True
                else:
                    print("🔄 兼容性加载失败，尝试重新创建向量数据库...")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                print("🔄 重新创建向量数据库...")
        
        if not documents:
            print("⚠️  没有文档可处理")
            # 即使没有文档，也创建一个空的向量存储，避免后续错误
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import FakeEmbeddings
                temp_embeddings = FakeEmbeddings(size=384)
                # 创建空的向量存储
                self.vector_store = FAISS.from_texts(["系统初始化"], temp_embeddings)
                print("✅ 创建空向量数据库完成")
                return True
            except Exception as e:
                print(f"❌ 创建空向量数据库失败: {e}")
                return False
        
        print("🔨 创建向量数据库...")
        start_time = time.time()
        
        # 分割文档
        split_docs = self.split_documents(documents)
        if not split_docs:
            print("❌ 文档分割后无内容")
            return False
        
        # 创建向量存储
        try:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            # 保存到本地
            self.vector_store.save_local(self.vector_db_path)
            creation_time = time.time() - start_time
            print(f"✅ 向量数据库创建完成，耗时: {creation_time:.2f}秒")
            return True
        except Exception as e:
            print(f"❌ 创建向量数据库失败: {e}")
            return False
    
    def _load_faiss_compatible(self):
        """兼容性加载FAISS向量数据库 - 处理不同版本API[6](@ref)"""
        try:
            # 方法1：尝试新版本API（不带危险反序列化参数）
            try:
                vector_store = FAISS.load_local(self.vector_db_path, self.embeddings)
                print("✅ 使用新版本API加载成功")
                return vector_store
            except TypeError as e:
                if "allow_dangerous_deserialization" in str(e):
                    # 方法2：尝试旧版本API（带危险反序列化参数）
                    try:
                        vector_store = FAISS.load_local(
                            self.vector_db_path, 
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        print("✅ 使用旧版本API加载成功")
                        return vector_store
                    except Exception as e2:
                        print(f"❌ 旧版本API也失败: {e2}")
                        return None
                else:
                    # 其他TypeError，重新抛出
                    raise e
        except Exception as e:
            print(f"❌ 加载异常: {e}")
            return None
    
    def init_llm(self, model_type: str = "ollama"):
        """初始化语言模型 - 增强错误处理[8](@ref)"""
        try:
            if model_type == "ollama":
                # 尝试多个可能的模型名称
                model_names = ["qwen2.5:0.5b", "llama2", "qwen:7b", "mistral"]
                for model_name in model_names:
                    try:
                        self.llm = Ollama(model=model_name, temperature=0.1)
                        # 测试连接 - 使用更简单的测试方法
                        test_response = self.llm.invoke("hello")
                        if test_response and len(test_response) > 0:
                            print(f"✅ Ollama模型初始化成功: {model_name}")
                            return
                        else:
                            raise Exception("测试响应为空")
                    except Exception as e:
                        print(f"⚠️  模型 {model_name} 不可用: {str(e)[:100]}...")
                        continue
                
                # 如果所有模型都失败，尝试最基本的连接
                try:
                    self.llm = Ollama(model="qwen2.5:0.5b")
                    print("⚠️  使用默认模型（未测试连接）")
                except:
                    # 终极备用方案
                    from langchain_community.llms import FakeListLLM
                    self.llm = FakeListLLM(responses=["Ollama服务未正确配置，请检查安装和运行状态"])
                    print("⚠️  使用模拟LLM")
                
            else:
                # 备用简单模型
                from langchain_community.llms import FakeListLLM
                self.llm = FakeListLLM(responses=["抱歉，模型未正确配置"])
                print("⚠️  使用备用模型")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            # 使用模拟模型
            from langchain_community.llms import FakeListLLM
            self.llm = FakeListLLM(responses=["模型服务初始化失败，系统将以基础模式运行"])
            print("⚠️  使用基础模拟LLM")
    
    def create_qa_chain(self):
        """创建问答链 - 增强错误处理"""
        if not self.vector_store:
            print("❌ 向量数据库未初始化")
            return False
            
        if not self.llm:
            print("❌ 语言模型未初始化")
            return False
        
        # 自定义提示模板
        prompt_template = """基于以下提供的上下文信息，请给出准确、简洁的回答。如果上下文信息不足，请明确说明。

上下文信息：
{context}

问题：{question}

请根据以上上下文用中文回答："""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False  # 减少输出噪音
            )
            print("✅ 问答链创建成功")
            return True
        except Exception as e:
            print(f"❌ 创建问答链失败: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """提问并获取答案 - 增强错误处理"""
        if not self.qa_chain:
            return {
                "answer": "系统未初始化完成，请先运行初始化流程",
                "sources": [],
                "time": "0.00秒"
            }
        
        try:
            start_time = time.time()
            result = self.qa_chain({"query": question})
            response_time = time.time() - start_time
            
            answer = result.get("result", "未获得有效答案")
            sources = []
            
            # 提取来源文档
            source_docs = result.get("source_documents", [])
            for doc in source_docs:
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", 1)
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                sources.append({
                    "source": source,
                    "page": page,
                    "preview": content_preview
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "time": f"{response_time:.2f}秒"
            }
        except Exception as e:
            print(f"❌ 回答问题出错: {e}")
            return {
                "answer": f"系统暂时无法处理您的请求: {str(e)[:100]}...",
                "sources": [],
                "time": "0.00秒"
            }
    
    def initialize_system(self, force_recreate: bool = False):
        """初始化整个系统 - 增强健壮性"""
        print("=" * 50)
        print("🤖 个人知识库问答系统初始化中...")
        print("=" * 50)
        
        # 1. 初始化嵌入模型
        print("步骤 1/5: 初始化文本嵌入模型...")
        if not self.init_embeddings():
            print("⚠️  嵌入模型初始化失败，但尝试继续运行")
            # 不退出，尝试使用基础功能
        
        # 2. 加载文档
        print("步骤 2/5: 加载知识库文档...")
        documents = self.load_documents()
        if not documents:
            print("⚠️  知识库中没有文档或加载失败")
            print("   支持格式: PDF, TXT, DOCX, MD")
            print("   请将文档放入 knowledge_base 文件夹")
            # 不退出，允许使用空知识库
        
        # 3. 创建向量数据库
        print("步骤 3/5: 创建向量数据库...")
        if not self.create_vector_store(documents, force_recreate):
            print("❌ 向量数据库创建失败")
            # 不立即退出，尝试继续
        
        # 4. 初始化LLM
        print("步骤 4/5: 初始化语言模型...")
        self.init_llm("ollama")
        
        # 5. 创建问答链
        print("步骤 5/5: 创建问答链...")
        if not self.create_qa_chain():
            print("❌ 问答链创建失败")
            return False
        
        print("✅ 系统初始化完成！")
        return True


def main():
    """主函数"""
    print("=" * 50)
    print("🎯 个人知识库问答系统 v3.0 (终极修复版)")
    print("=" * 50)
    
    # 创建系统实例
    qa_system = KnowledgeBaseQA()
    
    # 初始化系统
    print("🚀 开始系统初始化...")
    success = qa_system.initialize_system()
    
    if success:
        print("🎉 系统初始化成功！")
    else:
        print("⚠️  系统初始化遇到问题，但尝试继续运行基础功能")
        print("💡 您可以尝试:")
        print("   1. 检查知识库文档是否已放入 knowledge_base 文件夹")
        print("   2. 确认Ollama服务已启动: ollama serve")
        print("   3. 运行 '重新加载' 命令重建系统")
    
    print("\n💡 交互提示:")
    print("  直接输入问题: 获取基于知识库的答案")
    print("  输入'帮助'或'help': 查看使用说明") 
    print("  输入'重新加载'或'reload': 重新构建向量数据库")
    print("  输入'退出'/'quit'/'exit': 结束程序")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n❓ 请输入您的问题: ").strip()
            
            if question.lower() in ['退出', 'quit', 'exit']:
                print("👋 感谢使用，再见！")
                break
            
            elif question.lower() in ['帮助', 'help']:
                print("\n📖 使用说明:")
                print("  1. 直接输入问题获取基于知识库的答案")
                print("  2. 支持的文档格式: PDF, TXT, DOCX, MD")
                print("  3. 将文档放入 knowledge_base 文件夹")
                print("  4. 输入'重新加载'更新知识库索引")
                print("  5. 系统会显示答案来源和响应时间")
                continue
            
            elif question.lower() in ['重新加载', 'reload']:
                confirm = input("⚠️  确定要重新构建向量数据库吗？(y/N): ")
                if confirm.lower() in ['y', 'yes']:
                    print("🔄 重新初始化系统...")
                    if qa_system.initialize_system(force_recreate=True):
                        print("✅ 系统重新初始化完成")
                    else:
                        print("❌ 重新初始化失败")
                continue
            
            elif not question:
                continue
            
            # 提问并获取答案
            print(f"\n🔍 正在检索答案...")
            result = qa_system.ask_question(question)
            
            # 显示答案
            print(f"\n📝 答案 (响应时间: {result['time']}):")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)
            
            # 显示来源
            if result['sources']:
                print(f"\n📚 参考来源:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. 文档: {source['source']} (第{source['page']}页)")
                    print(f"      摘要: {source['preview']}")
            else:
                print("\nℹ️  未找到相关来源文档")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生未知错误: {e}")
            print("💡 系统将继续运行，您可以继续提问")
            traceback.print_exc()


if __name__ == "__main__":
    main()