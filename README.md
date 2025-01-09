[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/j1SuxzcN)
# Prompt Engineer 作業題目

## 作業內容

請使用 **LangChain** 套件完成以下作業，並實作提供的方法 `generate_hw01-04(question)`。實作於 **`student_assignment.py`** 中。
#### 創建 `.env` 文件（供學生使用）

在此作業中，您將需要在本地開發環境中設置必要的參數來支持程式運行。為了簡化環境變數的管理，請在項目根目錄下創建一個名為 `.env` 的文件，並在其中定義環境變數。

這個 `.env` 文件的主要用途是為了讓您能夠在自己的電腦上進行實作，並為`model_configurations.py`文件提供所需的參數。參加作業時，我們會提供具體的參數值供您填寫。以下是 .env 文件的範例格式：


```makefile
AZURE_OPENAI_GPT4O_ENDPOINT=your_endpoint_here
AZURE_OPENAI_GPT4O_KEY=your_api_key_here
AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT=your_deployment_name_here
AZURE_OPENAI_GPT4O_VERSION=your_api_version_here
```
#### 注意事項

- **請勿將 `.env` 文件上傳到任何版本控制系統（例如 GitHub）**，以避免洩漏敏感資訊。
- `.env` 文件僅供您在本地環境中使用，不需要提交作業時包含在內。

---

### 作業1

1. **問題**：`請回答台灣特定月份的紀念日有哪些(請用JSON格式呈現)?`
2. **範例**：`2024年台灣10月紀念日有哪些?`
3. **方法**：實作 `generate_hw01(question)`，用於回答上述問題。
4. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": [
             {
                 "date": "2024-10-10",
                 "name": "國慶日"
             }
        ]
     }
     ```

---

### 作業2

1. **問題**：`請回答台灣特定月份的紀念日有哪些(請用JSON格式呈現)?`
2. **範例**：`2024年台灣10月紀念日有哪些?`
3. **方法**：
   - 使用 Function Calling 的方式查詢指定的 API。
   - 實作 `generate_hw02(question)`，用於回答上述問題。
4. **指定 API**：
   - 使用 [Calendarific API](https://calendarific.com/)。
   - 步驟：
     1. 訪問 Calendarific 網站並註冊帳戶。
     2. 登錄後進入 Dashboard，取得您的 API Key。
5. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": [
             {
                 "date": "2024-10-10",
                 "name": "國慶日"
             },
             {
                 "date": "2024-10-09",
                 "name": "重陽節"
             },
             {
                 "date": "2024-10-21",
                 "name": "華僑節"
             },
             {
                 "date": "2024-10-25",
                 "name": "台灣光復節"
             },
             {
                 "date": "2024-10-31",
                 "name": "萬聖節"
             }
         ]
     }
     ```
     
---

### 作業3

1. **問題**：`根據 作業2 的回答，檢查某個節日是否包含在該月份的節日清單中，並回應是否需要新增該節日`
2. **範例**：`根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？`
3. **方法**：
   - 使用 RunnableWithMessageHistory 的方式記憶前一次的回答。
   - 實作 `generate_hw03(question2, question3)`，用於回答上述問題。 PS.question2是作業2的問題
4. **輸出格式**：
   - add : 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。
   - reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。
   - JSON 格式如下：
     ```json
     {
         "Result": 
             {
                 "add": true,
                 "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
             }
     }
     ```

---

### 作業4

1. **問題**：`請解析提供的圖片檔案 baseball.png，並回答圖片中有關問題的內容。`
2. **範例**：`請問中華台北的積分是多少`
3. **方法**：
   - 使用提供的圖片檔案 baseball.png 作為輸入數據來源，透過程式實現圖片內容的解析與問題的回答。
   - 實作 `generate_hw04(question)`，用於回答上述問題。
4. **輸出格式**：
   - JSON 格式如下：
     ```json
     {
         "Result": 
             {
                 "score": 5498
             }
     }
     ```

---

### 注意事項
- 必須使用 **LangChain** 套件完成方法實作。
- 確保輸出的格式與範例一致。

### 參考來源
- [作業1](https://python.langchain.com/docs/how_to/few_shot_examples_chat/)
- [作業2](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor)
- [作業2](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html)
- [作業3](https://python.langchain.com/docs/how_to/agent_executor/)
- [作業4](https://learn.microsoft.com/zh-tw/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest)

