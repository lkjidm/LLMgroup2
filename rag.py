import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 指定存放所有 PDF 的文件夹路径
KNOWLEDGE_FOLDER = "./knowledge"
VECTOR_DB_DIR = "./chroma_db"

def build_rag_from_folder():
    if not os.path.exists(KNOWLEDGE_FOLDER):
        os.makedirs(KNOWLEDGE_FOLDER)
        print(f"创建文件夹 {KNOWLEDGE_FOLDER} 成功，请将你的所有 PDF 手册放入该目录下！")
        return

    print(f"正在扫描并读取文件夹 '{KNOWLEDGE_FOLDER}' 中的所有 PDF 文件...")

    # 1. 使用 DirectoryLoader 批量加载目录下所有的 .pdf 文件
    # glob="./*.pdf" 表示只匹配 pdf 文件；loader_cls 指定底层使用 PyPDFLoader 解析
    loader = DirectoryLoader(
        KNOWLEDGE_FOLDER,
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True  # 显示读取进度条
    )
    
    pages = loader.load()
    print(f"成功加载文件夹内的所有 PDF，共计提取到 {len(pages)} 页文本。")

    if not pages:
        print("警告：文件夹内没有找到任何 PDF 文件！")
        return

    # 2. 统一切片（按 500 字拆分知识卡片）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(pages)
    print(f"文本切片完成，共生成 {len(docs)} 个知识片段。")

    # 3. 向量化并持久化保存至本地
    print("正在加载 Embedding 模型并创建向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"✅ 所有 PDF 知识已成功合并打包，保存至向量库: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_rag_from_folder()