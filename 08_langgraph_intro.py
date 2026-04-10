# ================================================
# AŞAMA 8 — LangGraph (Stateful Agent)
# ================================================
#
# LangChain'den LangGraph'a geçiş:
#
# LangChain AgentExecutor:
#   - Düz while loop
#   - State kaybolur, checkpoint yok
#   - Dallanma (if/else) kısıtlı
#
# LangGraph:
#   - Her adım "node" (düğüm)
#   - Adımlar arası "edge" (kenar)
#   - State her node'dan geçer, güncellenir
#   - Checkpoint: crash'te kaldığı yerden devam
#   - Human-in-the-loop kolay

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
import json
import datetime

# pip install langgraph
# pip install langchain-ollama

llm = ChatOllama(model="llama3", temperature=0)

# ── Tool'ları tanımla ──────────────────────────
@tool
def hesapla(ifade: str) -> str:
    """Matematiksel hesaplama yapar. Örnek: '25 * 4'"""
    try:
        return str(eval(ifade))
    except Exception as e:
        return f"Hata: {e}"

@tool
def saat_al() -> str:
    """Şu anki tarihi ve saati döner."""
    return datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

tools = [hesapla, saat_al]

# LLM'e tool'ları bağla (native tool calling)
llm_with_tools = llm.bind_tools(tools)

# ── State tanımla ──────────────────────────────
# Bu bizim "messages listesi" nin resmi hali.
# TypedDict: her alanın tipi belli.
# Annotated[list, operator.add]: yeni mesajlar listeye eklenir (replace değil)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ── Node'ları tanımla ──────────────────────────
def llm_node(state: AgentState) -> AgentState:
    """
    LLM Node: State'teki mesajları alır, LLM'e gönderir,
    cevabı state'e ekler.

    Bizim raw kodda: llm_output = call_ollama(messages)
    """
    cevap = llm_with_tools.invoke(state["messages"])
    return {"messages": [cevap]}

# ToolNode LangGraph'ın hazır tool çalıştırıcısı
# Bizim raw kodda: run_tool(tool_name, tool_args)
tool_node = ToolNode(tools)

# ── Graph oluştur ──────────────────────────────
print("=" * 40)
print("LangGraph agent oluşturuluyor...")
print("=" * 40)

graph = StateGraph(AgentState)

# Node'ları ekle
graph.add_node("llm",   llm_node)
graph.add_node("tools", tool_node)

# Başlangıç noktası
graph.set_entry_point("llm")

# Edge'ler (kenarlar) — akış kontrolü
# tools_condition: LLM tool çağırdı mı?
#   EVET → "tools" node'una git
#   HAYIR → END (bitir)
graph.add_conditional_edges(
    "llm",
    tools_condition,     # LangGraph'ın hazır karar fonksiyonu
)

# Tool çalıştıktan sonra tekrar LLM'e dön
graph.add_edge("tools", "llm")

# ── Checkpoint (memory) ────────────────────────
# MemorySaver: her adımı RAM'de saklar
# Production'da SqliteSaver veya PostgresSaver kullan
memory = MemorySaver()

# Graph'ı derle
app = graph.compile(checkpointer=memory)

# ── Test et ────────────────────────────────────
print("\n" + "=" * 40)
print("TEST 1: Basit hesaplama")
print("=" * 40)

config = {"configurable": {"thread_id": "test_1"}}

sonuc = app.invoke(
    {"messages": [HumanMessage(content="347 çarpı 23 kaç eder?")]},
    config=config
)
print(f"Cevap: {sonuc['messages'][-1].content}")

print("\n" + "=" * 40)
print("TEST 2: Saat sorusu")
print("=" * 40)

config2 = {"configurable": {"thread_id": "test_2"}}
sonuc2 = app.invoke(
    {"messages": [HumanMessage(content="Şu an saat kaç?")]},
    config=config2
)
print(f"Cevap: {sonuc2['messages'][-1].content}")

# ── Checkpoint testi — aynı thread devam eder ──
print("\n" + "=" * 40)
print("TEST 3: Checkpoint — konuşma devam ediyor")
print("=" * 40)

config3 = {"configurable": {"thread_id": "sohbet_1"}}

# İlk mesaj
app.invoke(
    {"messages": [HumanMessage(content="Merhaba, ben Emel.")]},
    config=config3
)

# Aynı thread — önceki konuşmayı hatırlıyor!
sonuc3 = app.invoke(
    {"messages": [HumanMessage(content="Adım ne benim?")]},
    config=config3
)
print(f"Cevap: {sonuc3['messages'][-1].content}")

# State'i incele
state = app.get_state(config3)
print(f"\nToplam mesaj sayısı: {len(state.values['messages'])}")

# ── Graph'ı görselleştir ───────────────────────
print("\n" + "=" * 40)
print("Graph yapısı:")
print("=" * 40)
print("""
  [START]
     ↓
  [LLM node]  ← mesajları işler
     ↓
  Tool çağrısı var mı?
    ↙           ↘
[tools node]  [END]
    ↓
  [LLM node]  ← tekrar
""")
