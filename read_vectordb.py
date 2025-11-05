import yaml
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction

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
    vectorstore_loaded = Milvus(
        embedding_function=embeddings,
        text_field=milvus_conf["semantic"]["text_field"],
        vector_field=milvus_conf["semantic"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
    )
elif mode == "bm25":
    print("使用全文檢索模式 (BM25 Retrieval)")
    builtin_conf = milvus_conf["bm25"]["builtin_function"]
    vectorstore_loaded = Milvus(
        embedding_function=None,
        builtin_function=BM25BuiltInFunction(**builtin_conf),
        text_field=milvus_conf["bm25"]["text_field"],
        vector_field=milvus_conf["bm25"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
    )
elif mode == "hybrid":
    print("使用混合檢索模式 (Hybrid Retrieval)")
    builtin_conf = milvus_conf["hybrid"]["builtin_function"]
    vectorstore_loaded = Milvus(
        embedding_function=embeddings,
        builtin_function=BM25BuiltInFunction(**builtin_conf),
        text_field=milvus_conf["hybrid"]["text_field"],
        vector_field=milvus_conf["hybrid"]["vector_field"],
        connection_args=milvus_conf["connection_args"],
        collection_name=milvus_conf["collection_name"],
    )
else:
    raise ValueError(f"Unknown Milvus mode: {mode}")

retriever_conf = config["retriever"]

query = "告訴我一些成語的典故？"
if mode == "hybrid" and "hybrid_search" in retriever_conf:
    similar_docs = vectorstore_loaded.similarity_search_with_score(
        query,
        k=retriever_conf["search_kwargs"]["k"],
        ranker_type=retriever_conf["hybrid_search"]["ranker_type"],
        ranker_params=retriever_conf["hybrid_search"]["ranker_params"],
        expr="directory == '4000-4999'"  # just an example filter
    )
else:
    similar_docs = vectorstore_loaded.similarity_search_with_score(
        query,
        k=retriever_conf["search_kwargs"]["k"],
        expr="directory == '4000-4999'"  # just an example filter
    )

for doc in similar_docs:
    print(doc)