# 🤖 Çok Etiketli Metin Sınıflandırma Web Uygulaması

Bu proje, eğitilmiş makine öğrenmesi modellerini kullanarak metinleri otomatik olarak kategorilere ayıran bir Flask web uygulamasıdır.

## 🎯 Özellikler

### 📊 Desteklenen Modeller
- **Logistic Regression** (MultiOutput)
- **Random Forest** (MultiOutput)
- **Classifier Chain (LR)** - Etiket bağımlılıklarını modeller
- **Classifier Chain (RF)**
- **Probabilistic Classifier Chain** - Belirsizlik modelleme
- **BERT** (DistilBERT) - Korelasyon-farkında loss ile

### 🏷️ Kategori Sınıflandırması
- **Hükümet & Sosyal**
- **Ekonomi & Finans**
- **Pazarlar & Ticaret**
- **İş & Endüstri**

### ✨ Web Arayüzü Özellikleri
- Modern ve responsive tasarım
- Gerçek zamanlı karakter sayacı
- Örnek metinler ile hızlı test
- Tüm modellerin karşılaştırmalı sonuçları
- Güven skoru grafikleri
- Sonuçları JSON/CSV formatında indirme
- API desteği

## 🚀 Kurulum

### 1. Gereksinimler
```bash
Python 3.8+
pip
```

⚠️ **ÖNEMLİ NOT**: Bu uygulamanın çalışabilmesi için bilgisayarınızda **numpy 2.0.2** veya uyumlu bir sürüm (1.26.4+) yüklü olmalıdır. Eski numpy sürümleri ile kaydedilmiş model dosyaları yüklenemeyebilir.

🚨 **DOSYA YOLU UYARISI**: Bu projeyi farklı bir bilgisayara kopyalarsanız veya farklı bir klasör yapısında çalıştırırsanız, aşağıdaki dosyalardaki **veri seti ve model dosya yollarını** güncellemek **zorunludur**:

- `app.py` - Model dosyalarının yükleme yolları
- `multi_model_classifier.py` - Veri seti ve model kaydetme yolları  
- Jupyter notebook dosyaları (`.ipynb`) - Veri seti okuma yolları

**Güncellenmesi gereken tipik yollar:**
- Veri seti yolu: `preprocessed_dataset_3.csv`
- Model klasörü: `saved_models/session_XXXXXX_XXXXXX/`
- Çıktı dosyaları ve görsel kaydetme yolları

Bu yolları güncellemeden çalıştırırsanız "FileNotFoundError" hatası alabilirsiniz.

### 2. Projeyi İndirin
```bash
git clone <proje-url>
cd Proje-Deneme13
```

### 3. Sanal Ortam Oluşturun
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 5. Modellerin Varlığını Kontrol Edin
Eğitilmiş modellerin `saved_models/session_20250822_134401/` klasöründe olduğundan emin olun:
```
saved_models/
└── session_20250822_134401/
    ├── tfidf_vectorizer.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── classifier_chain_lr.pkl
    ├── classifier_chain_rf.pkl
    ├── probabilistic_chain.pkl
    ├── bert_model.pkl
    ├── bert_tokenizer.pkl
    ├── label_columns.pkl
    ├── correlation_matrix.pkl
    ├── high_corr_pairs.pkl
    ├── performance_results.pkl
    └── model_parameters.json
```

### 6. Uygulamayı Başlatın
```bash
python app.py
```

Uygulama `http://localhost:5000` adresinde çalışmaya başlayacaktır.

## 🔧 Kullanım

### Web Arayüzü
1. Ana sayfada metin kutusuna sınıflandırılacak metni girin
2. "Sınıflandır" butonuna tıklayın
3. Tüm modellerin sonuçlarını karşılaştırın
4. İhtiyaç halinde sonuçları indirin

### API Kullanımı
```python
import requests

# POST /predict endpoint'i
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Sınıflandırılacak metin'})

results = response.json()
print(results)
```

### API Yanıt Formatı
```json
{
  "original_text": "Girilen metin",
  "processed_text": "İşlenmiş metin",
  "predictions": {
    "Logistic Regression": {
      "predicted_labels": ["Ekonomi & Finans"],
      "label_details": [
        {
          "label": "Ekonomi & Finans",
          "predicted": true,
          "probability": 0.85
        }
      ],
      "prediction_time": 15.2
    }
  },
  "summary": {
    "successful_models": 5,
    "total_models": 6,
    "consensus_labels": ["Ekonomi & Finans"],
    "most_predicted_labels": [["Ekonomi & Finans", 4]]
  }
}
```

## 📁 Proje Yapısı

```
Proje-Deneme13/
├── app.py                          # Ana Flask uygulaması
├── requirements.txt                # Python bağımlılıkları
├── README.md                      # Proje dokümantasyonu
├── templates/                     # HTML şablonları
│   ├── base.html                 # Ana şablon
│   ├── index.html                # Ana sayfa
│   ├── results.html              # Sonuçlar sayfası
│   ├── 404.html                  # 404 hata sayfası
│   └── 500.html                  # 500 hata sayfası
├── static/                       # Statik dosyalar
│   ├── css/
│   │   └── style.css            # Özel CSS stilleri
│   └── js/
│       └── main.js              # JavaScript fonksiyonları
├── saved_models/                 # Eğitilmiş modeller
│   └── session_20250822_134401/
│       ├── *.pkl               # Model dosyaları
│       └── *.json              # Konfigürasyon dosyaları
├── model-egitim-2.ipynb         # Model eğitim notebook'u
├── dataset.ipynb               # Veri analizi notebook'u
└── multi_model_classifier.py   # Model eğitim scripti
```

## 🧠 Metin Ön İşleme

Kullanıcıdan alınan metinler aşağıdaki adımlardan geçer:

1. **Küçük harfe çevirme** (`lower()`)
2. **Özel karakter temizleme** (sadece harf, rakam, boşluk)
3. **Fazla boşlukları normalize etme**
4. **Boş metinleri filtreleme**

Bu işlemler, modellerin eğitim sırasında gördüğü veri formatıyla tutarlılığı sağlar.

## 📊 Model Performansları

| Model | F1-Score | Doğruluk | Eğitim Süresi |
|-------|----------|----------|---------------|
| Logistic Regression | 0.925 | 0.840 | 5.8s |
| Probabilistic Chain | 0.915 | 0.841 | 390.8s |
| Random Forest | 0.890 | 0.783 | 17.5s |
| Classifier Chain (LR) | 0.914 | 0.840 | 6.4s |
| Classifier Chain (RF) | 0.865 | 0.745 | 17.1s |
| BERT | 0.738 | 0.664 | 747.8s |

## 🔒 Güvenlik

- Maksimum metin uzunluğu: 10,000 karakter
- Dosya yükleme boyut sınırı: 16MB
- CSRF koruması aktif
- Input sanitization uygulanmış

## 🐛 Hata Ayıklama

### Yaygın Sorunlar

1. **ModuleNotFoundError**: `pip install -r requirements.txt` komutunu çalıştırın
2. **Model dosyası bulunamadı**: `saved_models/` klasörünün doğru yerde olduğundan emin olun
3. **BERT hatası**: BERT modeli isteğe bağlıdır, diğer modeller çalışmaya devam eder
4. **Port hatası**: `app.py` dosyasında port numarasını değiştirin
5. **numpy._core hatası**: Numpy sürüm uyumsuzluğu. Aşağıdaki komutu çalıştırın:
   ```bash
   pip install numpy==1.26.4 pandas==2.2.0 scikit-learn==1.5.0 scipy==1.14.0 --upgrade
   ```

### Log Kontrolü
```bash
# Uygulama loglarını görüntüle
python app.py
```

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Sorularınız için issue oluşturun veya iletişime geçin.

---

**Not**: Bu uygulama eğitim ve araştırma amaçlı geliştirilmiştir. Üretim ortamında kullanmadan önce ek güvenlik testleri yapılması önerilir.

⚠️ **MODEL YOLU UYARISI**: Model klasör isimleri tarih/zaman damgası içerdiğinden, `app.py` dosyasındaki model yollarını güncel klasör ismiyle değiştirmeniz gerekir.

## 📥 Veri Seti ve Model Dosyaları

Projeyi çalıştırmak için gerekli veri seti ve model dosyalarını aşağıdaki Google Drive bağlantılarından indirebilirsiniz:

### 🗂️ İndirme Bağlantıları

- **Ham Veri Seti**: [Google Drive - Veri Seti](https://drive.google.com/drive/folders/186D2OXHlkDPWkKkJNQ990MA3GeSCZb_R?usp=drive_link)
- **Eğitilmiş Modeller**: [Google Drive - Modeller](https://drive.google.com/drive/folders/1-O_qRyaS8xXxf7xfUbaR4iimw90SLi5o?usp=drive_link)
- **İşlenmiş Veri Seti**: [Google Drive - Hazırlanmış Veri Seti](https://drive.google.com/file/d/1jGJGerFKQ8kJAXYoOrhJOi8SaYUbspfN/view?usp=drive_link)

⚠️ **ÖNEMLİ**: 
1. İndirdiğiniz model dosyalarını `saved_models/` klasörüne yerleştirin
2. Veri setini projenin ana dizinine yerleştirin
3. `app.py` dosyasındaki ilgili dosya yollarını güncelleyin

Bu dosyalar olmadan uygulama düzgün çalışmayacaktır. Lütfen kurulum adımlarını takip etmeden önce gerekli dosyaları indirdiğinizden emin olun.




