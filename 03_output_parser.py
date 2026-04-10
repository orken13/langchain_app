# ================================================
# AŞAMA 3 — Output Parser
# ================================================
#
# LLM her zaman string döner.
# Ama biz çoğu zaman yapılandırılmış veri isteriz:
#   liste, dict, JSON, Pydantic model...
#
# Output Parser → LLM çıktısını istediğin formata çevir.
# Bu özellikle agent'larda kritik:
# "tool parametrelerini JSON olarak ver" diyorsun.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
)
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

llm = ChatOllama(model="llama3", temperature=0)

# ── 1: StrOutputParser ─────────────────────────
print("=" * 40)
print("TEST 1: StrOutputParser")
print("=" * 40)
# LLM AIMessage döner, .content almak yerine
# direkt string almak için:

prompt = ChatPromptTemplate.from_messages([
    ("human", "{soru}")
])

# prompt | llm → AIMessage
# prompt | llm | StrOutputParser → string
chain = prompt | llm | StrOutputParser()

cevap = chain.invoke({"soru": "Python'un 3 avantajı nedir?"})
print(type(cevap))   # <class 'str'>
print(cevap)

# ── 2: JsonOutputParser ────────────────────────
print("\n" + "=" * 40)
print("TEST 2: JsonOutputParser")
print("=" * 40)

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sadece geçerli JSON döndür. Başka hiçbir şey yazma."),
    ("human",  "{dil} programlama dili hakkında JSON döndür. "
               "Alanlar: isim, yil, populer_kullanim (liste), zorluk (kolay/orta/zor)")
])

json_chain = json_prompt | llm | JsonOutputParser()

sonuc = json_chain.invoke({"dil": "Python"})
print(type(sonuc))   # <class 'dict'>
print(sonuc)
print(f"\nDil: {sonuc.get('isim')}")
print(f"Yıl: {sonuc.get('yil')}")

# ── 3: PydanticOutputParser ────────────────────
print("\n" + "=" * 40)
print("TEST 3: PydanticOutputParser (tip güvenli)")
print("=" * 40)

# Pydantic model tanımla — LLM bunu dolduracak
class UrunAnaliz(BaseModel):
    urun_adi:    str         = Field(description="Ürünün adı")
    fiyat_tl:    float       = Field(description="Tahmini TL fiyatı")
    kategori:    str         = Field(description="Ürün kategorisi")
    onerilen:    bool        = Field(description="Önerilir mi?")
    ozellikler:  list[str]   = Field(description="3 özellik listesi")

parser = PydanticOutputParser(pydantic_object=UrunAnaliz)

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ürün analizi yap. {format_instructions}"),
    ("human",  "{urun} hakkında analiz yap."),
]).partial(format_instructions=parser.get_format_instructions())

pydantic_chain = pydantic_prompt | llm | parser

sonuc = pydantic_chain.invoke({"urun": "iPhone 15"})
print(type(sonuc))           # <class 'UrunAnaliz'>
print(f"Ürün: {sonuc.urun_adi}")
print(f"Fiyat: {sonuc.fiyat_tl} TL")
print(f"Kategori: {sonuc.kategori}")
print(f"Önerilir: {sonuc.onerilen}")
print(f"Özellikler: {sonuc.ozellikler}")
