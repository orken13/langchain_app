# ================================================
# AŞAMA 2 — Prompt Template
# ================================================
#
# Raw kodda sistem prompt'u string olarak yazıyorduk:
#   SYSTEM_PROMPT = f"""Sen bir asistansın. {degisken}"""
#
# LangChain'de bunu şablonla yapıyoruz.
# Avantaj: değişkenleri güvenli inject et,
# tekrar kullanılabilir prompt'lar oluştur.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3", temperature=0)

# ── Temel Prompt Template ──────────────────────
print("=" * 40)
print("TEST 1: ChatPromptTemplate")
print("=" * 40)

# {konu} ve {seviye} değişken — runtime'da doldurulur
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımcı bir eğitim asistanısın. {seviye} seviyesinde açıkla."),
    ("human",  "{konu} nedir?"),
])

# Prompt'u doldur → messages listesine çevir
dolu_prompt = prompt.invoke({
    "konu":   "machine learning",
    "seviye": "başlangıç"
})

print("Oluşan mesajlar:")
for msg in dolu_prompt.messages:
    print(f"  [{msg.type}]: {msg.content}")

# LLM'e gönder
cevap = llm.invoke(dolu_prompt)
print(f"\nCevap: {cevap.content}")

# ── LCEL: Pipe operatörü | ─────────────────────
print("\n" + "=" * 40)
print("TEST 2: LCEL — prompt | llm zinciri")
print("=" * 40)

# LangChain'in en güçlü özelliği: | ile zincirleme
# prompt → llm → otomatik akış
chain = prompt | llm

cevap = chain.invoke({
    "konu":   "neural network",
    "seviye": "orta"
})
print(f"Cevap: {cevap.content}")

# ── Farklı şablonlar ───────────────────────────
print("\n" + "=" * 40)
print("TEST 3: Çeviri şablonu")
print("=" * 40)

ceviri_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir çevirmensin. Sadece çeviriyi yaz, açıklama ekleme."),
    ("human",  "Şu metni {kaynak_dil}'den {hedef_dil}'e çevir:\n\n{metin}"),
])

ceviri_chain = ceviri_prompt | llm

cevap = ceviri_chain.invoke({
    "kaynak_dil": "İngilizce",
    "hedef_dil":  "Türkçe",
    "metin":      "Artificial intelligence is transforming the world."
})
print(f"Çeviri: {cevap.content}")
