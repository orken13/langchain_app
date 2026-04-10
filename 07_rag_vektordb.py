# ================================================
# AŞAMA 7 — RAG + Vektör DB
# ================================================
#
# RAG = Retrieval Augmented Generation
# "LLM bilmiyorsa, önce DB'den bul, sonra cevapla"
#
# Önceki derste konuşmuştuk:
#   Vektör DB → anlam benzerliğine göre arama
#   Normal DB → kelime eşleşmesi
#
# Bu dosyada:
#   1. Metinleri vektöre çevir (embedding)
#   2. ChromaDB'ye kaydet
#   3. Soru gelince → alakalı metni bul
#   4. LLM'e "şu bağlamda cevapla" de

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

llm = ChatOllama(model="llama3", temperature=0)

# Embedding modeli — metni sayıya çeviren model
# Ollama'da mevcut embedding modeli
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Eğer nomic-embed-text yoksa: ollama pull nomic-embed-text

# ── 1: Dokümanları hazırla ─────────────────────
print("=" * 40)
print("Dokümanlar yükleniyor...")
print("=" * 40)

# Gerçek projede: PDF, web, CSV'den yüklersin
# Biz örnek olarak elle yazıyoruz
dokumanlar = [
    Document(
        page_content="""
        LangChain, büyük dil modelleri ile uygulama geliştirmeyi
        kolaylaştıran açık kaynak bir Python framework'üdür.
        2022 yılında Harrison Chase tarafından oluşturuldu.
        Prompt yönetimi, bellek, araç entegrasyonu sağlar.
        """,
        metadata={"kaynak": "langchain_intro", "sayfa": 1}
    ),
    Document(
        page_content="""
        LangGraph, LangChain ekosisteminin bir parçasıdır.
        Agent'ları yönlendirilen grafik (DAG) olarak modeller.
        State yönetimi, checkpoint ve human-in-the-loop desteği vardır.
        Karmaşık, çok adımlı agent'lar için idealdir.
        """,
        metadata={"kaynak": "langgraph_intro", "sayfa": 1}
    ),
    Document(
        page_content="""
        RAG (Retrieval Augmented Generation), LLM'lerin
        bilgi tabanını genişletmek için kullanılan bir tekniktir.
        Kullanıcı sorusu gelince önce vektör DB'de arama yapılır,
        bulunan belgeler LLM'e bağlam olarak verilir.
        Bu sayede LLM güncel ve domain-specific bilgiyle cevap verir.
        """,
        metadata={"kaynak": "rag_intro", "sayfa": 1}
    ),
    Document(
        page_content="""
        ChromaDB, açık kaynak bir vektör veritabanıdır.
        Python ile kolayca kullanılır, local çalışır.
        Metinleri embedding vektörlerine çevirip saklar.
        Cosine similarity ile semantik arama yapar.
        Ücretsizdir ve kurulumu basittir.
        """,
        metadata={"kaynak": "chromadb_intro", "sayfa": 1}
    ),
    Document(
        page_content="""
        Ollama, büyük dil modellerini local olarak çalıştırmaya
        yarayan açık kaynak bir araçtır. llama3, mistral, gemma
        gibi modelleri indirip offline kullanmana olanak tanır.
        API key gerektirmez, internet bağlantısına ihtiyaç duymaz.
        """,
        metadata={"kaynak": "ollama_intro", "sayfa": 1}
    ),
]

# ── 2: Metinleri parçala ───────────────────────
# Uzun metinler → küçük chunk'lara böl
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # her chunk max 200 karakter
    chunk_overlap=30,     # chunk'lar arası 30 karakter örtüşme
)
parca_dokumanlar = splitter.split_documents(dokumanlar)
print(f"Orijinal: {len(dokumanlar)} doküman")
print(f"Parçalanmış: {len(parca_dokumanlar)} chunk")

# ── 3: Vektör DB'ye kaydet ────────────────────
print("\nVektör DB oluşturuluyor...")
print("(İlk seferinde embedding hesaplanır, biraz sürer)")

vectordb = Chroma.from_documents(
    documents=parca_dokumanlar,
    embedding=embeddings,
    persist_directory="./chroma_db",  # diske kaydet
)
print(f"DB'ye {len(parca_dokumanlar)} chunk kaydedildi.")

# ── 4: Retriever ───────────────────────────────
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # en alakalı 2 chunk getir
)

# Retriever'ı test et
print("\n" + "=" * 40)
print("Retriever testi")
print("=" * 40)
test_sorgu = "LangGraph nedir?"
bulunan = retriever.invoke(test_sorgu)
print(f"Sorgu: {test_sorgu}")
print(f"Bulunan chunk sayısı: {len(bulunan)}")
for i, chunk in enumerate(bulunan):
    print(f"\nChunk {i+1} (kaynak: {chunk.metadata['kaynak']}):")
    print(chunk.page_content.strip())

# ── 5: RAG Chain ──────────────────────────────
print("\n" + "=" * 40)
print("RAG Chain testi")
print("=" * 40)

def format_docs(docs):
    """Bulunan dokümanları tek string'e birleştir."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen yardımcı bir asistansın.
Sadece aşağıdaki bağlamı kullanarak soruyu cevapla.
Bağlamda yoksa "Bu konuda bilgim yok" de.

BAĞLAM:
{context}"""),
    ("human", "{soru}"),
])

# RAG chain:
# soru → retriever (alakalı chunk'ları bul)
# soru + chunk'lar → LLM
# LLM → cevap
rag_chain = (
    {
        "context": retriever | format_docs,  # retriever çalışır, format_docs ile birleştirir
        "soru": RunnablePassthrough()        # soruyu olduğu gibi geçir
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Test soruları
sorular = [
    "LangGraph nedir ve ne işe yarar?",
    "ChromaDB nasıl çalışır?",
    "Ollama'yı neden kullanmalıyım?",
    "Python'da liste nasıl oluşturulur?",  # bağlamda yok
]

for soru in sorular:
    print(f"\nSoru: {soru}")
    cevap = rag_chain.invoke(soru)
    print(f"Cevap: {cevap}")
