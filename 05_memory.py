# ================================================
# AŞAMA 5 — Memory (Konuşma Hafızası)
# ================================================
#
# Raw kodda memory yoktu — her agent.py çalıştığında sıfırlanıyordu.
# LangChain'de memory = konuşma geçmişini otomatik yönet.
#
# 2024+ LangChain'de eski Memory sınıfları deprecated oldu.
# Yeni yol: mesaj geçmişini manuel veya
# RunnableWithMessageHistory ile yönet.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3", temperature=0.7)

# ── 1: Manuel mesaj geçmişi ────────────────────
print("=" * 40)
print("TEST 1: Manuel konuşma geçmişi")
print("=" * 40)

# Konuşma geçmişini elle yönet
# Bizim raw koddaki "messages" listesinin LangChain versiyonu
gecmis = []

def sohbet_et(kullanici_mesaji: str) -> str:
    gecmis.append(HumanMessage(content=kullanici_mesaji))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen yardımcı bir Türkçe asistansın."),
        MessagesPlaceholder(variable_name="gecmis"),  # ← geçmiş buraya giriyor
    ])

    chain = prompt | llm | StrOutputParser()
    cevap = chain.invoke({"gecmis": gecmis})

    gecmis.append(AIMessage(content=cevap))
    return cevap

# Çok turlu konuşma — agent önceki turu hatırlıyor mu?
c1 = sohbet_et("Benim adım Emel.")
print(f"Kullanıcı: Benim adım Emel.")
print(f"Agent: {c1}\n")

c2 = sohbet_et("Hangi şehirleri gezmeliyim Türkiye'de?")
print(f"Kullanıcı: Hangi şehirleri gezmeliyim?")
print(f"Agent: {c2}\n")

c3 = sohbet_et("Benim adım ne?")  # hatırlıyor mu?
print(f"Kullanıcı: Benim adım ne?")
print(f"Agent: {c3}")

print(f"\nGeçmişteki mesaj sayısı: {len(gecmis)}")

# ── 2: RunnableWithMessageHistory ─────────────
print("\n" + "=" * 40)
print("TEST 2: RunnableWithMessageHistory")
print("=" * 40)

# Her kullanıcı için ayrı session
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Session ID'ye göre konuşma geçmişini döner."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Chain oluştur
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir asistansın. Kullanıcıyı hatırla."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain2 = prompt2 | llm | StrOutputParser()

# Memory ile sar
with_memory = RunnableWithMessageHistory(
    chain2,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session A — Emel'in konuşması
config_a = {"configurable": {"session_id": "emel_session"}}

r1 = with_memory.invoke({"input": "Merhaba, ben Emel."}, config=config_a)
print(f"Emel: Merhaba, ben Emel.")
print(f"Agent: {r1}\n")

r2 = with_memory.invoke({"input": "Python öğreniyorum."}, config=config_a)
print(f"Emel: Python öğreniyorum.")
print(f"Agent: {r2}\n")

r3 = with_memory.invoke({"input": "Adım ne benim?"}, config=config_a)
print(f"Emel: Adım ne benim?")
print(f"Agent: {r3}\n")

# Session B — farklı kullanıcı, ayrı hafıza
config_b = {"configurable": {"session_id": "ali_session"}}
r4 = with_memory.invoke({"input": "Adım ne benim?"}, config=config_b)
print(f"Ali (farklı session): Adım ne benim?")
print(f"Agent: {r4}")  # Emel'i bilmemeli!

# Session içeriğini gör
print(f"\nEmel session mesaj sayısı: {len(session_store['emel_session'].messages)}")
print(f"Ali session mesaj sayısı:  {len(session_store['ali_session'].messages)}")
