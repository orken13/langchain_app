# ================================================
# AŞAMA 4 — Chain (LCEL)
# ================================================
#
# Chain = birden fazla adımı | ile birbirine bağlamak.
# LangChain Expression Language (LCEL) bu işi yapar.
#
# Bizim raw kodda her adımı elle çağırıyorduk:
#   llm_out = call_ollama(messages)
#   parsed  = parse_action(llm_out)
#   result  = run_tool(parsed)
#
# LCEL'de:
#   chain = prompt | llm | parser | tool
#   chain.invoke(input)

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

llm = ChatOllama(model="llama3", temperature=0)

# ── 1: Temel chain ─────────────────────────────
print("=" * 40)
print("TEST 1: Temel chain")
print("=" * 40)

ozet_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Sen bir özetleme asistanısın. Türkçe özetle."),
        ("human",  "Şunu {kelime_sayisi} kelimeyle özetle:\n\n{metin}")
    ])
    | llm
    | StrOutputParser()
)

metin = """
LangChain, büyük dil modelleri (LLM) ile uygulama geliştirmeyi
kolaylaştıran bir framework'tür. Prompt yönetimi, bellek, araç
entegrasyonu ve agent oluşturma gibi özellikleriyle geliştiricilere
güçlü bir ekosistem sunar.
"""

sonuc = ozet_chain.invoke({
    "kelime_sayisi": "10",
    "metin": metin
})
print(f"Özet: {sonuc}")

# ── 2: Sıralı chain (çıktı → giriş) ───────────
print("\n" + "=" * 40)
print("TEST 2: Sıralı chain — çevir sonra özetle")
print("=" * 40)

# Adım 1: İngilizce metni Türkçeye çevir
ceviri_prompt = ChatPromptTemplate.from_messages([
    ("system", "Metni Türkçeye çevir. Sadece çeviriyi yaz."),
    ("human",  "{metin}")
])
ceviri_chain = ceviri_prompt | llm | StrOutputParser()

# Adım 2: Çevrilmiş metni özetle
ozet_prompt = ChatPromptTemplate.from_messages([
    ("system", "Metni 1 cümleyle özetle."),
    ("human",  "{ceviri}")
])
ozet_chain2 = ozet_prompt | llm | StrOutputParser()

# İkisini birleştir: ilk chain'in çıktısı
# "ceviri" key'i ile ikinci chain'e gidiyor
tam_chain = (
    {"ceviri": ceviri_chain}   # ceviri_chain çıktısını "ceviri" key'ine koy
    | ozet_chain2
)

ingilizce = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
sonuc = tam_chain.invoke({"metin": ingilizce})
print(f"Sonuç: {sonuc}")

# ── 3: RunnablePassthrough — girişi geçir ──────
print("\n" + "=" * 40)
print("TEST 3: RunnablePassthrough")
print("=" * 40)

# Hem orijinal girişi hem de LLM cevabını bir arada tut
chain_with_input = (
    RunnablePassthrough.assign(
        cevap=ChatPromptTemplate.from_messages([
            ("human", "{soru}")
        ]) | llm | StrOutputParser()
    )
)

sonuc = chain_with_input.invoke({"soru": "Python nedir?"})
print(f"Soru:  {sonuc['soru']}")
print(f"Cevap: {sonuc['cevap']}")

# ── 4: RunnableLambda — custom fonksiyon ───────
print("\n" + "=" * 40)
print("TEST 4: RunnableLambda — kendi fonksiyonun")
print("=" * 40)

def buyuk_harf_yap(metin: str) -> str:
    """Chain içinde custom Python fonksiyonu kullanmak için."""
    return metin.upper()

chain_with_func = (
    ChatPromptTemplate.from_messages([("human", "{soru}")])
    | llm
    | StrOutputParser()
    | RunnableLambda(buyuk_harf_yap)   # custom fonksiyon
)

sonuc = chain_with_func.invoke({"soru": "merhaba de"})
print(f"Sonuç: {sonuc}")
