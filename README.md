# Irem's RAG Chatbot - Akbank GenAI Bootcamp
**Purpose:** Build a Retrieval-Augmented-Generation (RAG) chatbot that answers domain questions using documents & a large language model.
**Live Demo:** https://your-deploy-link

**Quick start**
1. git clone ...
2. python -m venv venv && source venv/bin/activate
3. pip install -r requirements.txt
4. python src/app.py

# RAG Tabanlı SQL Query Assistant
Akbank GenAI Bootcamp — RAG (Retrieval-Augmented Generation) ile doğal dilden SQL üreten asistan.

## Projenin Amacı
Bu proje, doğal dil sorularını veritabanı şemalarından (schema + örnek SQL) yararlanarak doğru SQL sorgularına çeviren bir RAG tabanlı asistan sunar. Kullanıcı sorusu:
1. Vektör arama ile en ilgili örnek/şema parçalarını getirir,
2. LLM (Google Gemini) ile bu bağlamı kullanıp final SQL'i üretir,
3. Üretilen SQL'i doğrular (format, güvenlik kuralları).

## Veri Seti
- Kullanılan veri seti: Hazır SQL/CREATE-QUESTION-ANSWER veri kümesi (örnek: HuggingFace SQL soru-örnek veri setleri).
- Projede orijinal veri seti yer almıyor — telif/izin gereksinimleri nedeniyle veri repoya dahil edilmedi.
- Veri işleme adımları: parçalara bölme (chunking), schema+question birleşimi, örnek SQL saklama.
- Eğer gated dataset kullanıyorsanız, ilgili erişim izinlerini almanız gerekir.

## Kullanılan Yöntemler / Teknolojiler
- Embedding: sentence-transformers (all-MiniLM-L6-v2)
- Vektör DB: FAISS (IndexFlatIP + L2-normalize)
- RAG Pipeline: Basit retrieval -> generate akışı (Retriever + SQLGenerator)
- Generation: Google Gemini (google-generativeai)
- Validation: sqlparse tabanlı format + güvenlik filtreleri (SELECT-only/forbidden list)
- Web UI: Streamlit (app.py)
- Test scriptleri: test_e2e.py, test_gemini.py

## Kurulum (Windows, venv)
1. Repo klonla ve venv oluştur:
   - python -m venv venv_rag
   - .\venv_rag\Scripts\activate
2. Bağımlılıkları yükle:
   - pip install -r requirements.txt
3. Ortam değişkenleri (.env):
   - Proje köküne `.env` oluştur (commit etmeyin)
     ```
     GEMINI_API_KEY=your_gemini_key_here
     OPENAI_API_KEY=your_openai_key_here  # (opsiyonel fallback)
     ```
   - Visual Studio Code terminal env yükleme sorunları için `python.terminal.useEnvFile=true` ayarını etkinleştirin veya `from dotenv import load_dotenv; load_dotenv()` ile manuel yükleyin.
4. (Opsiyonel) Embeddings oluştur (eğer embeddings.faiss yoksa):
   - python create_embeddings.py
   - Bu işlem için processed_chunks.json gereklidir (veya kendi veri pipeline'ınızı kullanın).
5. Hızlı testler:
   - Gemini bağlantısı: python test_gemini.py
   - End-to-end: python test_e2e.py

## Nasıl Çalıştırılır (Local)
- Konsoldan etkileşimli retrieval:
  - python retrieve.py  # interactive mode
  - veya komut satırı: python retrieve.py -q "How to find employees with salary above average?" -k 5
- Streamlit web arayüzü:
  - streamlit run app.py
  - Tarayıcıda http://localhost:8501 açılacak

## Deploy (Streamlit Cloud önerisi)
1. Kodlarınızı GitHub'a pushlayın.
2. Streamlit Cloud (https://share.streamlit.io) ile repoyu bağlayın.
3. Secrets / Environment variables bölümüne `GEMINI_API_KEY` ve diğer anahtarları ekleyin.
4. Deploy edildiğinde README sonuna deploy linkinizi ekleyin.

Deploy link: (buraya deploy URL'nizi ekleyin)

## Güvenlik & Doğrulama
- validate.py: Üretilen SQL'ler sqlparse ile formatlanır ve bir güvenlik filtresi uygulanır (varsayılan: destructive SQL keyword'ları engellenir).
- Öneri: Üretimi çalıştırmadan önce `validate_sql`'i SELECT-only modunda tutun. Gerçek veritabanına doğrudan sorgu çalıştırmayın.
- Açığa çıkan API anahtarlarını hemen iptal edin ve yenilerini güvenli şekilde saklayın (.env gitignore içinde).

## Proje Yapısı (özet)
- create_embeddings.py — chunks -> embeddings.faiss + chunks.pkl
- retrieve.py — FAISS + SentenceTransformer + Retriever sınıfı + interactive CLI
- generate.py — Gemini entegrasyonu + prompt oluşturma
- validate.py — SQL doğrulama
- test_gemini.py / test_e2e.py — bağlantı ve E2E testleri
- app.py — Streamlit web arayüzü
- README.md, requirements.txt, .env.example

## Örnek Kullanım
- Lokal test:
  - python test_e2e.py
  - streamlit run app.py
- API (isteğe bağlı): FastAPI backend oluşturup Bubble/Retool/Frontend ile entegre edebilirsiniz.

## Sonuçlar / Beklenen Çıktı
- Doğru yapılandırıldığında model soruya bağlı, bağlama uygun ve valid SQL üretir. (Örnek çıktı test_e2e.py içindeki demo ile kontrol edildi.)

## Geliştirilecek İyileştirmeler (işlerin notu)
- Reranking (BM25 + semantic score)
- Prompt mühendisliği: daha kesin SQL-only çıktısı
- Sandbox: SQLite readonly ile otomatik syntax çalıştırma (SELECT sorguları için)
- Unit testler (pytest) ve CI pipeline
- UI: daha interaktif Streamlit bileşenleri veya Bubble entegrasyonu (backend API ile)

## Lisans & Katkı
- Eğitim amaçlı. Katkılar için issue açabilirsiniz.

---

Hazırlayan: [Proje Sahibi]  
Not: .env dosyalarınızı kesinlikle commit etmeyin. Anahtarlarınızı hemen iptal edip yenileyin eğer eskileri açığa çıktıysa.
