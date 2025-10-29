import yaml
from dotenv import load_dotenv
import re
import json
from utils import RecursiveTextSplitterLite
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_milvus import Milvus, BM25BuiltInFunction

text_splitter = RecursiveTextSplitterLite(chunk_size=512, chunk_overlap=64)

total_chunks = []
with open("data/sampled_idiom.jsonl", "r", encoding="utf-8") as f:
    json_lines = f.readlines()
    for json_line in json_lines:
        json_obj = json.loads(json_line)
        (directory, filename), text = json_obj["filename"].split('/'), json_obj["text"]
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[\t\r\f -]+', ' ', text)
        text = text.strip()

        document = Document(page_content=text, metadata={"directory": directory, "filename": filename})
        chunks = text_splitter.split_documents([document])

        total_chunks.extend(chunks)

print(f"總共有 {len(total_chunks)} 個chunks")
for i, chunk in enumerate(total_chunks[:5]):
    print(f"\n第 {i} 個chunk：\n{chunk}\n")

load_dotenv()  # 讀取 .env

# 讀取 YAML 設定檔
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

milvus_conf = config["milvus"]
mode = milvus_conf.get("mode", "bm25")

# 建立 embeddings（語意或混合都需要）
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 根據模式建立 vectorstore
if mode == "semantic":
    print("使用語意檢索模式 (Semantic Retrieval)")
    vectorstore = Milvus.from_documents(
        documents=total_chunks,
        embedding=embeddings,
        text_field=milvus_conf["semantic"]["text_field"],
        vector_field=milvus_conf["semantic"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
        index_params=milvus_conf["semantic"]["index_params"],
        drop_old=False,
    )
elif mode == "bm25":
    print("使用全文檢索模式 (BM25 Retrieval)")
    builtin_conf = milvus_conf["bm25"]["builtin_function"]
    vectorstore = Milvus.from_documents(
        documents=total_chunks,
        embedding=None,
        builtin_function=BM25BuiltInFunction(**builtin_conf),
        text_field=milvus_conf["bm25"]["text_field"],
        vector_field=milvus_conf["bm25"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
        index_params=milvus_conf["bm25"]["index_params"],
        drop_old=False,
    )
elif mode == "hybrid":
    print("使用混合檢索模式 (Hybrid Retrieval)")
    builtin_conf = milvus_conf["hybrid"]["builtin_function"]
    vectorstore = Milvus.from_documents(
        documents=total_chunks,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(**builtin_conf),
        text_field=milvus_conf["hybrid"]["text_field"],
        vector_field=milvus_conf["hybrid"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
        index_params=milvus_conf["hybrid"]["index_params"],
        drop_old=False,
    )
else:
    raise ValueError(f"Unknown Milvus mode: {mode}")
