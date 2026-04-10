# LangChain Eğitimi — Temelden İleri Seviyeye
# Ollama (local) ile çalışır, API key gerekmez

## Kurulum

```bash
pip install langchain langchain-ollama langchain-community langgraph chromadb
```

## Ollama model hazırlık

```bash
ollama pull llama3
ollama pull nomic-embed-text   # RAG için embedding modeli
```

## Dosya sırası — bu sırayla çalıştır

| Dosya | Konu | Kavram |
|-------|------|--------|
| `01_llm_temel.py`      | LLM çağrısı         | invoke, stream, batch |
| `02_prompt_template.py`| Prompt şablonları   | ChatPromptTemplate, LCEL pipe |
| `03_output_parser.py`  | Çıktı biçimlendirme | StrOutputParser, JSON, Pydantic |
| `04_chain.py`          | Zincirleme          | LCEL, RunnableLambda, Passthrough |
| `05_memory.py`         | Konuşma hafızası    | MessageHistory, session |
| `06_tools_agent.py`    | Tool ve Agent       | @tool, create_react_agent |
| `07_rag_vektordb.py`   | RAG                 | ChromaDB, embedding, retriever |
| `08_langgraph_intro.py`| LangGraph           | StateGraph, node, edge, checkpoint |

## Raw kod ile karşılaştırma

```
Raw kodumuz          LangChain karşılığı
─────────────────────────────────────────
SYSTEM_PROMPT str    → ChatPromptTemplate
call_ollama()        → llm.invoke()
parse_action()       → create_react_agent (içinde)
run_tool()           → ToolNode (içinde)
while döngü          → AgentExecutor / LangGraph
messages listesi     → State / MessageHistory
context_manager.py   → MemorySaver / checkpoint
```

## Çalıştırma

```bash
python 01_llm_temel.py
python 02_prompt_template.py
# ... sırayla devam
```
