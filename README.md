# 个人知识库问答系统
- 一个基于 RAG 架构的本地化智能文档问答系统，能够将各类文档转换为可查询的知识库，并通过简洁的命令行界面进行语义问答，保障数据隐私的同时提供准确、可追溯的答案。
- 本项目给出的示例文档是本人PDF格式的高数和C语言笔记。

## 核心特性
- **多格式支持**：PDF、TXT、DOCX、MD 等  
- **智能检索**：基于 FAISS 向量数据库实现高效语义检索  
- **本地化部署**：支持 Ollama 本地模型，确保数据不离开本地环境  

## 主要功能
- **文档智能处理**：自动解析、分割文档，并构建向量索引  
- **语义检索**：基于向量相似度匹配，精准定位相关信息片段  
- **智能问答**：结合大语言模型生成准确、可靠的答案  
- **来源追溯**：显示答案所在的文档及具体位置，提升答案可信度  
- **交互式界面**：提供简洁易用的命令行交互体验  

## 环境要求
- **操作系统**：Windows 10/11  
- **Python 版本**：3.8 或更高版本  
- **Anaconda**  

## 安装步骤
### 1. 克隆项目
- git clone https://github.com/B612sandboxes/Q_A-system.git
- cd "个人知识库问答系统"
### 2. 创建虚拟环境（推荐）
- **创建虚拟环境** python -m venv rag_system_fixed
- **选择解释器** ctrl+shift+P选择python解释器，选择conda里的rag_system
- **激活虚拟环境**（Windows）conda activate rag_system
### 3. 安装依赖 
- pip install -r requirements.txt
### 4. 配置 Ollama
- **安装 Ollama**：https://ollama.com/
- **拉取推荐模型**：ollama pull qwen2.5:0.5b
- **启动服务（后台运行）**：ollama serve
### 5. 准备知识库文档
- 将您的文档（PDF、TXT、DOCX 等）放入项目目录下的 `knowledge_base` 文件夹中。

## 运行系统
- cd "个人知识库问答系统"
- python app.py

## 使用方法
### 1. 提问
在提示符后输入您的问题，例如：高等数学的主要内容是什么？ C语言中指针是什么？
### 2. 交互命令
- 输入 `退出` 或 `quit` 结束程序  
- 输入 `重新加载` 或 `reload` 重新构建知识库索引（更新文档后使用）  
- 输入 `帮助` 或 `help` 查看使用说明  

## 技术栈
- **后端框架**：[LangChain](https://github.com/langchain-ai/langchain)  
- **向量数据库**：[FAISS](https://github.com/facebookresearch/faiss)（Facebook AI 相似性搜索）  
- **文本嵌入**：[sentence-transformers](https://github.com/UKPLab/sentence-transformers)  
- **大语言模型**：[Ollama](https://ollama.com/)（本地部署）  
- **文档解析**：PyPDF2、python-docx 等  

## 致谢
- 感谢 [LangChain](https://github.com/langchain-ai/langchain) 框架提供的强大支持  
- 感谢 [Ollama](https://ollama.com/) 项目提供的便捷本地大模型运行环境
