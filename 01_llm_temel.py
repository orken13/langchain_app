# ================================================
# AŞAMA 1 — LLM Temel Kullanım
# ================================================
#
# LangChain'de her şey "LLM nesnesi" üzerinden gider.
# .invoke() → tek seferlik çağrı
# .stream() → token token akış
#
# Bizim raw kodda:
#   requests.post(ollama_url, json=payload)
# LangChain'de:
#   llm.invoke("soru")
# Aynı şey — LangChain sadece bunu sarmallıyor.

from langchain_ollama import ChatOllama

# ── Model oluştur ──────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.7,   # 0 = deterministik, 1 = yaratıcı
)

print("=" * 40)
print("TEST 1: Basit invoke")
print("=" * 40)

response = llm.invoke("Python nedir? 2 cümleyle anlat.")

# response bir AIMessage objesi
print(type(response))          # <class 'AIMessage'>
print(response.content)        # asıl metin burda
print(response.response_metadata)  # token sayısı vs.

print("\n" + "=" * 40)
print("TEST 2: Stream (token token)")
print("=" * 40)

# Streaming — her token gelince ekrana yaz
for chunk in llm.stream("Yapay zeka nedir? Kısaca anlat."):
    print(chunk.content, end="", flush=True)
print()  # satır sonu

print("\n" + "=" * 40)
print("TEST 3: Batch (aynı anda birden fazla)")
print("=" * 40)

# Birden fazla soruyu aynı anda gönder
sorular = [
    "Python nedir?",
    "LangChain nedir?",
    "Agent nedir?",
]
cevaplar = llm.batch(sorular)
for soru, cevap in zip(sorular, cevaplar):
    print(f"\nSoru: {soru}")
    print(f"Cevap: {cevap.content[:100]}...")
