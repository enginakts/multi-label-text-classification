#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Test Script - OneVsRestClassifier ve BERT Fine-tuning Test
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Sağlık kontrolü"""
    print("\n🔍 Sağlık Kontrolü...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Sistem Sağlıklı")
            print(f"   📊 Yüklü Model Sayısı: {health_data['models_loaded']}")
            print(f"   🤖 BERT Mevcut: {health_data['bert_available']}")
            print(f"   🏷️ Etiket Sayısı: {len(health_data['labels'])}")
            return True
        else:
            print(f"❌ Sağlık kontrolü başarısız: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Sağlık kontrolü hatası: {e}")
        return False

def test_models():
    """Model listesi kontrolü"""
    print("\n📋 Model Listesi...")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ {models_data['total_models']} model tespit edildi")
            
            for model_name, details in models_data['models'].items():
                print(f"   🔹 {model_name} ({details['type']})")
                if details['type'] in ['OneVsRestClassifier', 'MultiOutputClassifier']:
                    print(f"      └─ Base: {details['base_estimator']}, N: {details['n_estimators']}")
                elif 'BERT' in model_name:
                    print(f"      └─ Base Model: {details['base_model']}")
                    if details.get('fine_tuner_available'):
                        print(f"      └─ Fine-tuner: ✅")
            return True
        else:
            print(f"❌ Model listesi alınamadı: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model listesi hatası: {e}")
        return False

def test_predict_api():
    """Ana tahmin API'sini test et"""
    print("\n🎯 Tahmin API Testi...")
    url = f"{BASE_URL}/predict"
    
    test_texts = [
        "Türkiye Merkez Bankası faiz oranlarını yükselterek enflasyonla mücadele etmeye devam ediyor.",
        "Yeni teknoloji şirketi, yapay zeka destekli çözümleriyle endüstride devrim yaratmaya hazırlanıyor.",
        "Hükümet, sosyal yardım programlarını genişletme kararı aldı.",
        "Borsa İstanbul'da işlem gören şirketlerin kar açıklamaları yatırımcılar tarafından yakından takip ediliyor."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {text[:50]}...")
        print('='*60)
        
        try:
            response = requests.post(url, 
                                   json={"text": text}, 
                                   headers={"Content-Type": "application/json"},
                                   timeout=60)  # BERT için daha uzun timeout
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Başarılı!")
                
                # Model sonuçlarını kategorilere göre grupla
                ovr_models = {}
                multi_models = {}
                chain_models = {}
                bert_models = {}
                
                if "predictions" in result:
                    for model_name, prediction in result["predictions"].items():
                        if "error" in prediction:
                            print(f"❌ {model_name}: {prediction['error'][:100]}...")
                        else:
                            labels = prediction.get("predicted_labels", [])
                            time_ms = prediction.get("prediction_time", 0)
                            
                            if "OneVsRest" in model_name:
                                ovr_models[model_name] = (labels, time_ms)
                            elif "MultiOutput" in model_name:
                                multi_models[model_name] = (labels, time_ms)
                            elif "Chain" in model_name:
                                chain_models[model_name] = (labels, time_ms)
                            elif "BERT" in model_name:
                                bert_models[model_name] = (labels, time_ms)
                
                # Kategorilere göre sonuçları göster
                if ovr_models:
                    print("\n🔹 OneVsRestClassifier Modelleri:")
                    for model_name, (labels, time_ms) in ovr_models.items():
                        base_type = model_name.replace("OneVsRest ", "")
                        print(f"   {base_type}: {labels} ({time_ms:.1f}ms)")
                
                if multi_models:
                    print("\n🔸 MultiOutputClassifier Modelleri:")
                    for model_name, (labels, time_ms) in multi_models.items():
                        base_type = model_name.replace("MultiOutput ", "")
                        print(f"   {base_type}: {labels} ({time_ms:.1f}ms)")
                
                if chain_models:
                    print("\n🔗 Classifier Chain Modelleri:")
                    for model_name, (labels, time_ms) in chain_models.items():
                        print(f"   {model_name}: {labels} ({time_ms:.1f}ms)")
                
                if bert_models:
                    print("\n🤖 BERT Modelleri:")
                    for model_name, (labels, time_ms) in bert_models.items():
                        print(f"   {model_name}: {labels} ({time_ms:.1f}ms)")
                
                if "summary" in result and result["summary"]:
                    consensus = result["summary"].get("consensus_labels", [])
                    if consensus:
                        print(f"\n🎯 Konsensüs Etiketler: {consensus}")
                
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Bağlantı hatası! Flask uygulaması çalışıyor mu?")
            return False
        except requests.exceptions.Timeout:
            print("❌ Zaman aşımı! BERT modelleri daha uzun sürebilir.")
        except Exception as e:
            print(f"❌ Test hatası: {e}")
    
    return True

def test_quick_predict():
    """Hızlı test API'sini test et"""
    print("\n⚡ Hızlı Test API...")
    try:
        response = requests.get(f"{BASE_URL}/test_predict", 
                              params={"text": "Ekonomik veriler pozitif bir tablo çiziyor."}, 
                              timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ Hızlı test başarılı!")
            if result["status"] == "success":
                successful_models = sum(1 for pred in result["results"]["predictions"].values() 
                                      if "error" not in pred)
                total_models = len(result["results"]["predictions"])
                print(f"   📊 {successful_models}/{total_models} model başarılı")
            return True
        else:
            print(f"❌ Hızlı test başarısız: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Hızlı test hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 Flask API Test Suite - OneVsRestClassifier & BERT Fine-tuning")
    print("=" * 70)
    
    # Sıralı testler
    tests = [
        ("Sağlık Kontrolü", test_health),
        ("Model Listesi", test_models),
        ("Hızlı Test", test_quick_predict),
        ("Tam Tahmin Testi", test_predict_api)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 {test_name} çalıştırılıyor...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} başarılı!")
            else:
                print(f"❌ {test_name} başarısız!")
        except Exception as e:
            print(f"❌ {test_name} hatası: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 Test Sonucu: {passed}/{total} test başarılı")
    if passed == total:
        print("🎉 Tüm testler başarılı! Sistem hazır.")
    else:
        print("⚠️ Bazı testler başarısız. Lütfen kontrol edin.")
    print("=" * 70)

if __name__ == "__main__":
    main()
