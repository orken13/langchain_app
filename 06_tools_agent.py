# ================================================
# AŞAMA 6 — Tools ve Agent
# ================================================
#
# Burası kritik! Bizim raw ReAct loop'umuzu
# LangChain nasıl yapıyor göreceğiz.
#
# Raw kodumuzda:
#   TOOL_SCHEMAS = [...]   → tool tanımları
#   run_tool(name, args)   → tool çalıştırıcı
#   parse_action(text)     → LLM çıktısını parse et
#   while loop             → döngü
#
# LangChain'de:
#   @tool decorator        → tool tanımla
#   create_react_agent()   → loop otomatik
#   AgentExecutor          → çalıştır

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate
import datetime
import json

llm = ChatOllama(model="llama3", temperature=0)

# ── Tool tanımla — @tool decorator ─────────────
# Docstring çok önemli! LLM bunu okuyarak
# "hangi tool'u ne zaman kullanayım?" kararı verir.

@tool
def hesap_makinesi(ifade: str) -> str:
    """
    Matematiksel işlem yapar.
    Kullanım: '2 + 2', '10 * 5', '100 / 4' gibi ifadeler.
    """
    try:
        sonuc = eval(ifade)
        return f"Sonuç: {sonuc}"
    except Exception as e:
        return f"Hata: {e}"

@tool
def guncel_saat() -> str:
    """
    Şu anki tarih ve saati döner.
    Saat veya tarih sorulduğunda kullan.
    """
    simdi = datetime.datetime.now()
    return simdi.strftime("%d.%m.%Y %H:%M:%S")

@tool
def kelime_sayar(metin: str) -> str:
    """
    Bir metindeki kelime, karakter ve cümle sayısını döner.
    Metin analizi için kullan.
    """
    kelimeler  = len(metin.split())
    karakterler = len(metin)
    cumleler   = metin.count('.') + metin.count('!') + metin.count('?')
    return json.dumps({
        "kelime_sayisi":    kelimeler,
        "karakter_sayisi":  karakterler,
        "cumle_sayisi":     cumleler
    }, ensure_ascii=False)

tools = [hesap_makinesi, guncel_saat, kelime_sayar]

# ── Tool'u direkt test et ───────────────────────
print("=" * 40)
print("Tool direkt test")
print("=" * 40)
print(hesap_makinesi.invoke({"ifade": "15 * 8"}))
print(guncel_saat.invoke({}))
print(kelime_sayar.invoke({"metin": "Merhaba dünya. Bu bir test."}))

# ── Agent oluştur ──────────────────────────────
print("\n" + "=" * 40)
print("Agent oluşturuluyor...")
print("=" * 40)

# ReAct prompt — LLM'e "Düşünce/Aksiyon/Gözlem" formatını öğretir
# Bizim SYSTEM_PROMPT'umuza benzer ama LangChain formatında
react_prompt = PromptTemplate.from_template("""
Sen yardımcı bir Türkçe asistansın. Soruları cevaplamak için araçları kullan.

Kullanabileceğin araçlar:
{tools}

Araç isimleri: {tool_names}

Çalışma formatın:
Düşünce: [Ne yapmalıyım?]
Aksiyon: [araç_adı]
Aksiyon Girdisi: [araç parametresi]
Gözlem: [araç sonucu]
... (tekrar et)
Düşünce: Artık cevabı biliyorum.
Final Cevap: [Türkçe net cevap]

Başla!

Soru: {input}
{agent_scratchpad}
""")

# Agent oluştur — bizim parse_action + loop mantığını içeriyor
agent = create_react_agent(llm, tools, react_prompt)

# AgentExecutor = bizim while döngümüz
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # her adımı ekrana yaz (debug için)
    max_iterations=5,       # bizim MAX_ITERATIONS
    handle_parsing_errors=True,  # parse hatalarında crash etme
)

# ── Soruları test et ───────────────────────────
sorular = [
    "235 çarpı 17 kaçtır?",
    "Şu an saat kaç?",
    "Bu cümledeki kelime sayısı kaç: 'LangChain agent geliştirme öğreniyorum'",
]

for soru in sorular:
    print(f"\n{'='*40}")
    print(f"SORU: {soru}")
    print("="*40)
    try:
        sonuc = executor.invoke({"input": soru})
        print(f"\nFINAL: {sonuc['output']}")
    except Exception as e:
        print(f"Hata: {e}")
