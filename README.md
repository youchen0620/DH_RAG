# DH-RAG

這是一個使用 Gemini、LangChain 以及 Milvus-Lite 的專案，旨在利用檢索增強生成（RAG）技術，從文本資料中提取相關內容並生成回答。

## 功能

* **Milvus Lite**：使用輕量級的向量資料庫 Milvus Lite 來存儲和檢索文本向量。
* **資料處理**：更新、讀取和刪除向量資料庫中的資料。
* **多種檢索模式**：支持語意檢索、BM25 全文檢索和混合檢索模式。
* **檢索增強生成 (RAG)**：結合了檢索（從知識庫中查找相關內容）和生成（由 LLM 產生回答）的技術。
* **可配置性**：通過 `config.yaml` 文件，可以輕鬆配置模型、資料庫和檢索參數。

## 安裝

1.  Clone this repository：
    ```bash
    git clone https://github.com/youchen0620/DH_RAG.git
    cd DH_RAG
    ```

2.  安裝所需的 Python 庫：
    ```bash
    pip install -r requirements.txt
    ```

3.  創建一個 `.env` 文件，並在其中設置您的 Google API 密鑰，請參考 `.env.example`：
    ```
    GOOGLE_API_KEY="your_google_api_key"
    ```

## 使用

1.  **更新向量資料庫**：
    運行 `update_vectordb.py` 將 `data/${your_data}.jsonl` 中的資料加載到 Milvus Lite 資料庫中。
    ```bash
    python update_vectordb.py
    ```

2.  **執行 RAG 查詢**：
    運行 `RAG.py` 對資料庫執行 RAG 查詢。
    ```bash
    python RAG.py
    ```
    您可以在 `RAG.py` 中修改查詢內容。

3.  **讀取向量資料庫**：
    運行 `read_vectordb.py` 讀取資料庫中的內容。
    ```bash
    python read_vectordb.py
    ```

4.  **刪除向量資料庫**：
    運行 `delete_vectordb.py` 刪除資料庫中的內容。
    ```bash
    python delete_vectordb.py
    ```

## 配置

所有配置都在 `config.yaml` 文件中進行。

*   **google**：配置 Gemini 模型的參數，如模型名稱、溫度等。
*   **milvus**：配置 Milvus Lite 資料庫的參數，如模式（`semantic`、`bm25` 或 `hybrid`）、連接參數和集合名稱。
*   **retriever**：配置檢索器的參數，如 `k`（返回的文檔數）。
*   **prompt**：配置提示模板。
