# Irem's RAGent - Akbank GenAI Bootcamp
## Proje Amacı 
Bu proje, SQL ve veritabanı şemalarına dayalı doğal dil sorgularını anlayarak doğru SQL sorgularını üreten bir Retrieval-Augmented Generation (RAG) tabanlı asistan geliştirmeyi amaçlamaktadır.

Amaç, LLM’lerin SQL oluştururken sıklıkla yaşadığı tablo veya sütun isimlerini uydurma (hallucination) problemini ortadan kaldırmak ve gerçek şema bilgisiyle temellendirilmiş, güvenilir SQL sorguları üretmektir.

Kullanıcı doğal dilde bir sorgu (örneğin “1000’den büyük nüfusa sahip şehirlerin isimlerini getir”) girdiğinde sistem, veritabanı şemasını içeren CREATE TABLE ifadeleri arasından en ilgili olanları bulur, bu bilgileri bağlam olarak LLM’e iletir ve doğru SQL sorgusunu oluşturur.
 
 Kullanıcı sorusu:
1. Vektör arama ile en ilgili örnek/şema parçalarını getirir,
2. LLM (Google Gemini) ile bu bağlamı kullanıp final SQL'i üretir,
3. Üretilen SQL'i doğrular (format, güvenlik kuralları).

## Veri Seti
- Kullanılan veri seti: Bu projede kullanılan veri seti, WikiSQL ve Spider veri kümelerinden türetilmiş birleştirilmiş ve temizlenmiş bir versiyondur.
- Veri seti Hugging Face üzerinde çekilmiştir ve LLM tabanlı text-to-SQL modellerinin hallucination hatalarını azaltmayı hedefler. Hazır SQL/CREATE-QUESTION-ANSWER veri kümesidir
- @misc{b-mc2_2023_sql-create-context,

title   = {sql-create-context Dataset},

author  = {b-mc2}, 
  
  year    = {2023},
  
  url     = https://huggingface.co/datasets/b-mc2/sql-create-context
  
  note    = {This dataset was created by modifying data from the following sources: \cite{zhongSeq2SQL2017, yu2018spider}.},}.
- Veri işleme adımları parçalara bölme (chunking), schema+question birleşimi, örnek SQL saklamadan oluşur.
  
- Veri Setinin Özellikleri:
1. Toplam 78,577 örnek (Doğal dil sorguları /İlgili CREATE TABLE ifadeleri/ Sorgunun doğru SQL çıktısı)
2. Her örnekte, kullanıcı sorusu (question), tablo şeması (context), ve doğru sorgu (answer) alanları yer alır.
3. Gerçek veri satırları (rows) yerine yalnızca şema bilgileri (CREATE TABLE) kullanılır.
Bu, LLM’lerin şema bilgisine dayalı SQL üretme becerilerini ölçmek için idealdir.

## Çözüm Mimarisi
1️. Veri Alma 

Kaynak dosyalar: SQL şemaları (CREATE TABLE ifadeleri)
Her dosya okunur, gereksiz karakterlerden temizlenir ve küçük parçalara (chunks) bölünür.

2️. Embedding 

Her chunk, SentenceTransformer modeli ile embedding’e dönüştürülür.
Bu embedding’ler, sorguların anlam bazlı benzerliğini ölçmekte kullanılır.

(OpenAI embedding modeli kullanılmadı, yalnızca SentenceTransformer tercih edildi.)

3. Vektör Veritabanı (Vector DB)

Embedding’ler FAISS vektör veritabanında saklanır.
Kullanıcının sorgusu embedding’e dönüştürülür ve en benzer k parçalar (top_k) geri getirilir.

4. Retriever Katmanı

FAISS içinden gelen en benzer context parçaları alınır.
Bu parçalar, bağlamsal yanıt oluşturmak için birleştirilir.

5.  Context Birleştirme (Context Assembly)

Elde edilen şemalar, prompt template içinde birleştirilerek LLM modeline gönderilir.
Böylece model, yalnızca ilgili tablo/sütun isimlerini kullanarak SQL üretir.

6. Yanıt Üretimi (Generator)

Gemini API (Google Generative AI) kullanılarak son yanıt üretilir.
Yanıtlar genellikle SQL sorgusu formatında döndürülür.

7️. Web Arayüzü (UI)

Web arayüzü Streamlit ile geliştirilmiştir.
Kullanıcı doğal dil sorgusunu girer → model yanıt üretir → yanıt ve kullanılan kaynaklar (ör. tablo adı, chunk id) arayüzde gösterilir.

## Kullanılan Yöntemler / Teknolojiler
- Embedding: sentence-transformers (all-MiniLM-L6-v2)
- Vektör DB: FAISS (IndexFlatIP + L2-normalize)
- RAG Pipeline: Basit retrieval -> generate akışı (Retriever + SQLGenerator)
- Generation: Google Gemini (google-generativeai)
- Validation: sqlparse tabanlı format + güvenlik filtreleri (SELECT-only/forbidden list)
- Web UI: Streamlit (app.py)

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

## Nasıl Çalıştırılır (Local)
- Konsoldan etkileşimli retrieval:
  - python retrieve.py  # interactive mode
  - veya komut satırı: python retrieve.py -q "How to find employees with salary above average?" -k 3
- Streamlit web arayüzü:
  - streamlit run app.py
  - Tarayıcıda http://localhost:**** açılacak


Deploy link: https://huggingface.co/spaces/iremrit/iremrit-s_rag_chatbot_gemini


## Proje Yapısı (özet)
- create_embeddings.py — chunks -> embeddings.faiss + chunks.pkl
- app.py:
— FAISS + SentenceTransformer + Retriever sınıfı + interactive CLI
— Gemini entegrasyonu + prompt oluşturma
— SQL doğrulama
— Streamlit web arayüzü
- README.md, requirements.txt, .env.example

## Örnek Kullanım
  - streamlit run app.py
- API (isteğe bağlı): FastAPI backend oluşturup Bubble/Retool/Frontend ile entegre edebilirsiniz.

## Sonuçlar / Beklenen Çıktı
- Doğru yapılandırıldığında model soruya bağlı, bağlama uygun ve valid SQL üretir. 
## Geliştirilecek İyileştirmeler (işlerin notu)
- Reranking (BM25 + semantic score)
- Prompt mühendisliği: daha kesin SQL-only çıktısı
- Sandbox: SQLite readonly ile otomatik syntax çalıştırma (SELECT sorguları için)
- Unit testler (pytest) ve CI pipeline
- UI: daha interaktif Streamlit bileşenleri veya Bubble entegrasyonu (backend API ile)



Hazırlayan: RANA IREM TURHAN  

