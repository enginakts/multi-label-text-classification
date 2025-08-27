#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Uygulaması - Çok Etiketli Metin Sınıflandırma
Eğitilmiş modelleri kullanarak kullanıcı metinlerini sınıflandırır
"""

import os
import pickle
import json
import re
import time
import warnings
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
import numpy as np

# Sklearn kütüphaneleri
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# BERT için gerekli kütüphaneler
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torch.optim import AdamW
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT kütüphaneleri bulunamadı. Sadece sklearn modelleri kullanılacak.")

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'multi_label_classifier_secret_key_2024'

# Konfigürasyon
MODEL_DIR = 'C:/Users/engin/Desktop/Multi-label text classification/saved_models/session_2_20250825_062911/'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global değişkenler
models = {}
vectorizer = None
label_columns = []
model_info = {}

class ProbabilisticClassifierChain(BaseEstimator, ClassifierMixin):
    """Probabilistic Classifier Chain - belirsizlik modelleme ile"""
    
    def __init__(self, base_estimator, order=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        
    def fit(self, X, Y):
        """Modeli eğit"""
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)
        
        self.classes_ = []
        self.estimators_ = []
        self.n_labels_ = Y.shape[1]
        
        # Etiket sırasını belirle
        if self.order is None:
            # Korelasyon bazlı sıralama
            corr_matrix = np.corrcoef(Y.T)
            # En yüksek ortalama korelasyona sahip etiketleri önce al
            mean_corr = np.nanmean(np.abs(corr_matrix), axis=1)
            self.order_ = np.argsort(-mean_corr)
        else:
            self.order_ = self.order
            
        # Her etiket için bir estimator eğit
        for i, label_idx in enumerate(self.order_):
            y = Y[:, label_idx]
            self.classes_.append(unique_labels(y))
            
            # Önceki etiketlerin tahminlerini özellik olarak ekle
            if i == 0:
                X_extended = X
            else:
                # Önceki etiketlerin gerçek değerlerini kullan (eğitim sırasında)
                prev_labels = Y[:, self.order_[:i]]
                if hasattr(X, 'toarray'):
                    X_extended = np.hstack([X.toarray(), prev_labels])
                else:
                    X_extended = np.hstack([X, prev_labels])
            
            # Estimator'ı kopyala ve eğit
            estimator = pickle.loads(pickle.dumps(self.base_estimator))
            estimator.fit(X_extended, y)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        """Tahmin yap"""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)
        
        n_samples = X.shape[0]
        Y_pred = np.zeros((n_samples, self.n_labels_))
        
        # Sıralı tahmin
        for i, (label_idx, estimator) in enumerate(zip(self.order_, self.estimators_)):
            if i == 0:
                X_extended = X
            else:
                # Önceki tahminleri özellik olarak kullan
                prev_preds = Y_pred[:, self.order_[:i]]
                if hasattr(X, 'toarray'):
                    X_extended = np.hstack([X.toarray(), prev_preds])
                else:
                    X_extended = np.hstack([X, prev_preds])
            
            Y_pred[:, label_idx] = estimator.predict(X_extended)
            
        return Y_pred.astype(int)
    
    def predict_proba(self, X):
        """Olasılık tahminleri"""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)
        
        n_samples = X.shape[0]
        Y_proba = np.zeros((n_samples, self.n_labels_))
        
        for i, (label_idx, estimator) in enumerate(zip(self.order_, self.estimators_)):
            if i == 0:
                X_extended = X
            else:
                # Monte Carlo sampling ile belirsizlik modelle
                prev_preds = Y_proba[:, self.order_[:i]]
                if hasattr(X, 'toarray'):
                    X_extended = np.hstack([X.toarray(), prev_preds])
                else:
                    X_extended = np.hstack([X, prev_preds])
            
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_extended)
                if proba.shape[1] == 2:
                    Y_proba[:, label_idx] = proba[:, 1]
                else:
                    Y_proba[:, label_idx] = proba[:, 0]
            else:
                # Eğer predict_proba yoksa decision_function kullan
                if hasattr(estimator, 'decision_function'):
                    scores = estimator.decision_function(X_extended)
                    Y_proba[:, label_idx] = 1 / (1 + np.exp(-scores))  # Sigmoid
                else:
                    Y_proba[:, label_idx] = estimator.predict(X_extended)
                    
        return Y_proba

# BERT Model Sınıfları ve Fine-tuning (Flask uygulaması için)
if BERT_AVAILABLE:
    # Özel Dataset sınıfı
    class MultiLabelDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }
    
    # Gelişmiş BERT Multi-Label Classifier
    class BERTMultiLabelClassifier(nn.Module):
        def __init__(self, model_name='distilbert-base-uncased', num_labels=49, dropout_rate=0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout_rate)
            # Çoklu katmanlı classifier
            hidden_size = self.bert.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size // 2, num_labels)
            )
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token + mean pooling
            cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
            mean_output = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling
            combined = (cls_output + mean_output) / 2  # Ortalama al
            
            output = self.dropout(combined)
            logits = self.classifier(output)
            return self.sigmoid(logits)
    
    # Fine-tuning Trainer sınıfı (Flask için basitleştirilmiş)
    class BERTFineTuner:
        def __init__(self, model_name='distilbert-base-uncased', num_labels=49, device=None):
            self.model_name = model_name
            self.num_labels = num_labels
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.tokenizer = None
        
        def predict(self, texts, batch_size=32):
            """Tahmin yap"""
            if self.model is None:
                raise ValueError("Model henüz yüklenmedi!")
            
            dataset = MultiLabelDataset(texts, [[0] * self.num_labels] * len(texts), self.tokenizer)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            self.model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    probs = outputs.cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    predictions.extend(preds)
                    probabilities.extend(probs)
            
            return np.array(predictions), np.array(probabilities)

def text_preprocessing(text):
    """Metin ön işleme fonksiyonu - eğitim sırasında kullanılanla aynı"""
    if pd.notna(text) and text:
        # Küçük harfe çevir ve özel karakterleri temizle
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
        # Fazla boşlukları temizle
        return ' '.join(cleaned.split())
    else:
        return ""

def load_models():
    """Eğitilmiş modelleri yükle"""
    global models, vectorizer, label_columns, model_info
    
    print("🔄 Modeller yükleniyor...")
    
    try:
        # Model parametrelerini yükle
        with open(os.path.join(MODEL_DIR, 'model_parameters.json'), 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Label columns'ı yükle
        with open(os.path.join(MODEL_DIR, 'label_columns.pkl'), 'rb') as f:
            label_columns = pickle.load(f)
        
        # TF-IDF Vectorizer'ı yükle
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Sklearn modellerini yükle
        model_files = {
            'OneVsRest Logistic Regression': 'ovr_logistic_regression.pkl',
            'MultiOutput Logistic Regression': 'multi_logistic_regression.pkl',
            'OneVsRest SVM': 'ovr_svm.pkl',
            'OneVsRest Random Forest': 'ovr_random_forest.pkl',
            'MultiOutput Random Forest': 'multi_random_forest.pkl',
            'Classifier Chain (LR)': 'classifier_chain_lr.pkl',
            'Classifier Chain (RF)': 'classifier_chain_rf.pkl',
            'Probabilistic Chain': 'probabilistic_chain.pkl'
        }
        
        # SVM Feature Selector'ı yükle (eğer varsa)
        svm_feature_selector = None
        try:
            with open(os.path.join(MODEL_DIR, 'svm_feature_selector.pkl'), 'rb') as f:
                svm_feature_selector = pickle.load(f)
            models['svm_feature_selector'] = svm_feature_selector
            print("✅ SVM Feature Selector yüklendi")
        except FileNotFoundError:
            print("ℹ️ SVM Feature Selector bulunamadı (opsiyonel)")
        except Exception as e:
            print(f"⚠️ SVM Feature Selector yükleme hatası: {e}")
        
        for model_name, filename in model_files.items():
            try:
                with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"✅ {model_name} yüklendi")
            except Exception as e:
                print(f"❌ {model_name} yüklenemedi: {e}")
                import traceback
                print(f"   Detaylı hata: {traceback.format_exc()}")
        
        # BERT Fine-tuned modelini yükle (eğer varsa)
        if BERT_AVAILABLE:
            try:
                with open(os.path.join(MODEL_DIR, 'bert_finetuned_model.pkl'), 'rb') as f:
                    models['BERT'] = pickle.load(f)
                with open(os.path.join(MODEL_DIR, 'bert_finetuned_tokenizer.pkl'), 'rb') as f:
                    models['BERT_tokenizer'] = pickle.load(f)
                # BERT Fine-tuner'ı da yükle (eğer varsa)
                try:
                    with open(os.path.join(MODEL_DIR, 'bert_fine_tuner.pkl'), 'rb') as f:
                        models['BERT_fine_tuner'] = pickle.load(f)
                    print("✅ BERT Fine-tuned modeli ve tuner yüklendi")
                except:
                    print("✅ BERT Fine-tuned modeli yüklendi (tuner olmadan)")
            except Exception as e:
                print(f"❌ BERT Fine-tuned modeli yüklenemedi: {e}")
                import traceback
                print(f"   Detaylı hata: {traceback.format_exc()}")
        
        # Performans sonuçlarını yükle
        try:
            with open(os.path.join(MODEL_DIR, 'performance_results.pkl'), 'rb') as f:
                model_info['performance'] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Performans sonuçları yüklenemedi: {e}")
            import traceback
            print(f"   Detaylı hata: {traceback.format_exc()}")
            
        print(f"🎯 Toplam {len(models)} model yüklendi")
        print(f"📊 Etiket sayısı: {len(label_columns)}")
        print(f"🏷️ Etiketler: {', '.join(label_columns)}")
        
    except Exception as e:
        print(f"💥 Model yükleme hatası: {e}")
        import traceback
        print(f"   Detaylı hata: {traceback.format_exc()}")
        return False
    
    return True

def predict_with_models(text):
    """Tüm modellerle tahmin yap"""
    if not text or not text.strip():
        return {"error": "Boş metin"}
    
    # Metin ön işleme
    processed_text = text_preprocessing(text)
    if not processed_text:
        return {"error": "İşlenebilir metin bulunamadı"}
    
    # TF-IDF vektörleştirme
    try:
        text_tfidf = vectorizer.transform([processed_text])
    except Exception as e:
        return {"error": f"Vektörleştirme hatası: {str(e)}"}
    
    results = {
        "original_text": text,
        "processed_text": processed_text,
        "predictions": {},
        "probabilities": {},
        "summary": {}
    }
    
    # Sklearn modelleriyle tahmin
    for model_name, model in models.items():
        if model_name in ['BERT', 'BERT_tokenizer', 'BERT_fine_tuner', 'svm_feature_selector']:
            continue
        
        # OneVsRest ve MultiOutput modellerini özel olarak işle
        if 'OneVsRest' in model_name or 'MultiOutput' in model_name:
            try:
                start_time = time.time()
                
                # SVM için özel işleme (feature selector varsa)
                if 'SVM' in model_name and 'svm_feature_selector' in models:
                    text_tfidf_input = models['svm_feature_selector'].transform(text_tfidf)
                else:
                    text_tfidf_input = text_tfidf
                
                # Tahmin yap
                prediction = model.predict(text_tfidf_input)
                if len(prediction.shape) > 1:
                    prediction = prediction[0]
                
                # Olasılık tahmini (OneVsRest ve MultiOutput için)
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        # OneVsRest için özel olasılık hesaplama
                        if 'OneVsRest' in model_name:
                            proba_list = []
                            for i, estimator in enumerate(model.estimators_):
                                try:
                                    if hasattr(estimator, 'predict_proba'):
                                        proba = estimator.predict_proba(text_tfidf_input)
                                        if proba.shape[1] == 2:
                                            proba_list.append(proba[0, 1])  # Positive class probability
                                        else:
                                            proba_list.append(proba[0, 0])
                                    elif hasattr(estimator, 'decision_function') and 'SVM' in model_name:
                                        # SVM için decision function kullan ve sigmoid ile çevir
                                        decision = estimator.decision_function(text_tfidf_input)
                                        prob = 1 / (1 + np.exp(-decision[0]))
                                        proba_list.append(prob)
                                    else:
                                        proba_list.append(0.5)  # Default
                                except:
                                    proba_list.append(0.5)  # Default probability
                            probabilities = proba_list
                        else:
                            # MultiOutput için standart olasılık hesaplama
                            proba = model.predict_proba(text_tfidf)
                            if len(proba.shape) > 1:
                                proba = proba[0]
                            if isinstance(proba, list) and len(proba) > 0 and hasattr(proba[0], 'shape'):
                                # Multi-output durumu
                                probabilities = [prob[1] if len(prob) > 1 else prob[0] for prob in proba]
                            else:
                                probabilities = proba.tolist() if hasattr(proba, 'tolist') else proba
                    except Exception as prob_error:
                        print(f"   {model_name} predict_proba hatası: {prob_error}")
                        probabilities = [0.5] * len(label_columns)
                
                prediction_time = time.time() - start_time
                
                # Sonuçları hazırla
                predicted_labels = []
                label_details = []
                
                # Tahmin dizisi uzunluğunu kontrol et
                if len(prediction) != len(label_columns):
                    print(f"   ⚠️ {model_name}: Tahmin uzunluğu ({len(prediction)}) etiket sayısından ({len(label_columns)}) farklı")
                    if len(prediction) < len(label_columns):
                        prediction = list(prediction) + [0] * (len(label_columns) - len(prediction))
                    else:
                        prediction = prediction[:len(label_columns)]
                
                for i, (label, pred) in enumerate(zip(label_columns, prediction)):
                    if pred == 1:
                        predicted_labels.append(label)
                    
                    prob = probabilities[i] if probabilities and i < len(probabilities) else None
                    label_details.append({
                        "label": label,
                        "predicted": bool(pred),
                        "probability": float(prob) if prob is not None else None
                    })
                
                results["predictions"][model_name] = {
                    "predicted_labels": predicted_labels,
                    "label_details": label_details,
                    "prediction_time": round(prediction_time * 1000, 2)
                }
                continue
                
            except Exception as e:
                import traceback
                error_details = {
                    "error": f"OneVsRest/MultiOutput model hatası: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                results["predictions"][model_name] = error_details
                print(f"❌ {model_name} tahmin hatası: {e}")
                continue
        
        # ClassifierChain modellerini özel olarak işle (sklearn versiyon uyumsuzluğu için)
        if 'Chain' in model_name and model_name != 'Probabilistic Chain':
            try:
                start_time = time.time()
                
                # Manuel ClassifierChain tahmin (sklearn versiyon uyumsuzluğu nedeniyle)
                if hasattr(model, 'estimators_') and hasattr(model, 'order_'):
                    predictions = np.zeros(len(label_columns))
                    probabilities = []
                    
                    # Her estimator için sıralı tahmin
                    for i, (label_idx, estimator) in enumerate(zip(model.order_, model.estimators_)):
                        if i == 0:
                            X_extended = text_tfidf
                        else:
                            # Önceki tahminleri ekle
                            prev_preds = predictions[model.order_[:i]].reshape(1, -1)
                            X_extended = np.hstack([text_tfidf.toarray(), prev_preds])
                        
                        # Tahmin yap
                        pred = estimator.predict(X_extended)[0]
                        predictions[label_idx] = pred
                        
                        # Olasılık hesapla
                        if hasattr(estimator, 'predict_proba'):
                            try:
                                proba = estimator.predict_proba(X_extended)
                                if proba.shape[1] > 1:
                                    prob_value = proba[0, 1]
                                else:
                                    prob_value = proba[0, 0]
                            except:
                                prob_value = 0.5
                        else:
                            prob_value = 0.5
                        
                        probabilities.append(prob_value)
                    
                    # Olasılıkları doğru sırayla düzenle
                    ordered_probs = [0.0] * len(label_columns)
                    for i, label_idx in enumerate(model.order_):
                        ordered_probs[label_idx] = probabilities[i]
                    
                    prediction = predictions
                    probabilities = ordered_probs
                    
                else:
                    # Fallback: boş tahmin
                    prediction = np.zeros(len(label_columns))
                    probabilities = [0.5] * len(label_columns)
                
                prediction_time = time.time() - start_time
                
                # Sonuçları hazırla
                predicted_labels = []
                label_details = []
                
                for i, (label, pred, prob) in enumerate(zip(label_columns, prediction, probabilities)):
                    if pred == 1:
                        predicted_labels.append(label)
                    
                    label_details.append({
                        "label": label,
                        "predicted": bool(pred),
                        "probability": float(prob) if prob is not None else None
                    })
                
                results["predictions"][model_name] = {
                    "predicted_labels": predicted_labels,
                    "label_details": label_details,
                    "prediction_time": round(prediction_time * 1000, 2)
                }
                continue
                
            except Exception as e:
                import traceback
                error_details = {
                    "error": f"Sklearn versiyon uyumsuzluğu: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                results["predictions"][model_name] = error_details
                print(f"❌ {model_name} ClassifierChain hatası: {e}")
                print(f"   Detaylı hata: {traceback.format_exc()}")
                continue
            
        try:
            start_time = time.time()
            
            # Binary tahmin - güvenli şekilde
            try:
                prediction = model.predict(text_tfidf)
                if len(prediction.shape) > 1:
                    prediction = prediction[0]
                elif prediction.shape[0] == 0:
                    raise ValueError("Model boş tahmin döndürdü")
            except Exception as pred_error:
                print(f"   {model_name} predict hatası: {pred_error}")
                raise pred_error
            
            # Olasılık tahmini (eğer destekleniyorsa)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(text_tfidf)
                    if len(proba.shape) > 1:
                        proba = proba[0]
                    if len(proba.shape) == 2:  # Multi-output durumu
                        probabilities = [prob[1] if len(prob) > 1 else prob[0] for prob in proba]
                    else:
                        probabilities = proba.tolist()
                except Exception as prob_error:
                    print(f"   {model_name} predict_proba hatası: {prob_error}")
                    probabilities = None
            elif hasattr(model, 'decision_function'):
                try:
                    decision = model.decision_function(text_tfidf)
                    if len(decision.shape) > 1:
                        decision = decision[0]
                    # Sigmoid fonksiyonu ile olasılığa çevir
                    probabilities = [1 / (1 + np.exp(-score)) for score in decision]
                except Exception as dec_error:
                    print(f"   {model_name} decision_function hatası: {dec_error}")
                    probabilities = None
            
            # ClassifierChain özel durumu
            if 'Chain' in model_name and probabilities is None:
                try:
                    # ClassifierChain için manuel olasılık hesaplama
                    if hasattr(model, 'estimators_'):
                        probabilities = []
                        for estimator in model.estimators_:
                            if hasattr(estimator, 'predict_proba'):
                                # Her estimator için ayrı tahmin yap
                                est_proba = estimator.predict_proba(text_tfidf)
                                if len(est_proba.shape) > 1 and est_proba.shape[1] > 1:
                                    probabilities.append(est_proba[0, 1])
                                else:
                                    probabilities.append(est_proba[0])
                            else:
                                probabilities.append(0.5)  # Default
                except Exception as chain_error:
                    print(f"   {model_name} chain olasılık hatası: {chain_error}")
                    probabilities = None
            
            prediction_time = time.time() - start_time
            
            # Sonuçları hazırla - güvenli indeksleme
            predicted_labels = []
            label_details = []
            
            # Tahmin dizisi uzunluğunu kontrol et
            if len(prediction) != len(label_columns):
                print(f"   ⚠️ {model_name}: Tahmin uzunluğu ({len(prediction)}) etiket sayısından ({len(label_columns)}) farklı")
                # Eksik tahminleri 0 ile doldur veya fazlaları kes
                if len(prediction) < len(label_columns):
                    prediction = list(prediction) + [0] * (len(label_columns) - len(prediction))
                else:
                    prediction = prediction[:len(label_columns)]
            
            for i, (label, pred) in enumerate(zip(label_columns, prediction)):
                if pred == 1:
                    predicted_labels.append(label)
                
                prob = probabilities[i] if probabilities and i < len(probabilities) else None
                label_details.append({
                    "label": label,
                    "predicted": bool(pred),
                    "probability": float(prob) if prob is not None else None
                })
            
            results["predictions"][model_name] = {
                "predicted_labels": predicted_labels,
                "label_details": label_details,
                "prediction_time": round(prediction_time * 1000, 2)  # ms
            }
            
        except Exception as e:
            import traceback
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            results["predictions"][model_name] = error_details
            print(f"❌ {model_name} tahmin hatası: {e}")
            print(f"   Detaylı hata: {traceback.format_exc()}")
    
    # BERT Fine-tuned modeli ile tahmin (eğer varsa)
    if BERT_AVAILABLE and 'BERT' in models and 'BERT_tokenizer' in models:
        try:
            start_time = time.time()
            
            bert_model = models['BERT']
            tokenizer = models['BERT_tokenizer']
            
            # Eğer fine-tuner varsa onu kullan
            if 'BERT_fine_tuner' in models:
                fine_tuner = models['BERT_fine_tuner']
                try:
                    # Fine-tuner'ın model ve tokenizer'ını ayarla
                    fine_tuner.model = bert_model
                    fine_tuner.tokenizer = tokenizer
                    fine_tuner.num_labels = len(label_columns)
                    
                    predictions, probabilities = fine_tuner.predict([processed_text], batch_size=1)
                    predictions = predictions[0]
                    probabilities = probabilities[0]
                except Exception as fine_tuner_error:
                    print(f"Fine-tuner hatası, manuel tahmine geçiliyor: {fine_tuner_error}")
                    # Manuel tahmin
                    encoding = tokenizer(
                        processed_text,
                        truncation=True,
                        padding=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                    
                    bert_model.eval()
                    with torch.no_grad():
                        outputs = bert_model(encoding['input_ids'], encoding['attention_mask'])
                        probabilities = outputs.cpu().numpy()[0]
                        predictions = (probabilities > 0.5).astype(int)
            else:
                # Manuel tahmin
                encoding = tokenizer(
                    processed_text,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                bert_model.eval()
                with torch.no_grad():
                    outputs = bert_model(encoding['input_ids'], encoding['attention_mask'])
                    probabilities = outputs.cpu().numpy()[0]
                    predictions = (probabilities > 0.5).astype(int)
            
            prediction_time = time.time() - start_time
            
            # Sonuçları hazırla
            predicted_labels = []
            label_details = []
            
            for i, (label, pred, prob) in enumerate(zip(label_columns, predictions, probabilities)):
                if pred == 1:
                    predicted_labels.append(label)
                
                label_details.append({
                    "label": label,
                    "predicted": bool(pred),
                    "probability": float(prob)
                })
            
            results["predictions"]["BERT Fine-tuning"] = {
                "predicted_labels": predicted_labels,
                "label_details": label_details,
                "prediction_time": round(prediction_time * 1000, 2)
            }
            
        except Exception as e:
            import traceback
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            results["predictions"]["BERT Fine-tuning"] = error_details
            print(f"❌ BERT Fine-tuning tahmin hatası: {e}")
            print(f"   Detaylı hata: {traceback.format_exc()}")
    
    # Özet istatistikler
    successful_models = [name for name, result in results["predictions"].items() 
                        if "error" not in result]
    
    if successful_models:
        # En çok tahmin edilen etiketler
        all_predictions = {}
        for model_name in successful_models:
            for label in results["predictions"][model_name]["predicted_labels"]:
                all_predictions[label] = all_predictions.get(label, 0) + 1
        
        results["summary"] = {
            "successful_models": len(successful_models),
            "total_models": len(results["predictions"]),
            "most_predicted_labels": sorted(all_predictions.items(), 
                                          key=lambda x: x[1], reverse=True),
            "consensus_labels": [label for label, count in all_predictions.items() 
                               if count >= len(successful_models) // 2 + 1]
        }
    
    return results

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html', 
                         models=list(models.keys()), 
                         labels=label_columns,
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin API endpoint'i"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Metin girilmedi"}), 400
        
        if len(text) > 10000:  # Maksimum metin uzunluğu
            return jsonify({"error": "Metin çok uzun (max 10000 karakter)"}), 400
        
        results = predict_with_models(text)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Form ile tahmin"""
    text = request.form.get('text', '').strip()
    
    if not text:
        flash('Lütfen bir metin girin!', 'error')
        return render_template('index.html', models=list(models.keys()), 
                             labels=label_columns, model_info=model_info)
    
    try:
        results = predict_with_models(text)
        return render_template('results.html', results=results, 
                             models=list(models.keys()), labels=label_columns)
    except Exception as e:
        flash(f'Tahmin hatası: {str(e)}', 'error')
        return render_template('index.html', models=list(models.keys()), 
                             labels=label_columns, model_info=model_info)

@app.route('/health')
def health():
    """Sağlık kontrolü"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "labels": label_columns,
        "bert_available": BERT_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/models')
def list_models():
    """Yüklü modelleri listele"""
    model_details = {}
    for model_name, model in models.items():
        if model_name in ['BERT', 'BERT_tokenizer', 'BERT_fine_tuner']:
            continue
        
        model_type = "Unknown"
        if 'OneVsRest' in model_name:
            model_type = "OneVsRestClassifier"
        elif 'MultiOutput' in model_name:
            model_type = "MultiOutputClassifier"
        elif 'Chain' in model_name:
            model_type = "ClassifierChain"
        
        model_details[model_name] = {
            "type": model_type,
            "base_estimator": str(type(getattr(model, 'estimator', getattr(model, 'estimators_', [None])[0] if hasattr(model, 'estimators_') else None)).__name__) if hasattr(model, 'estimator') or hasattr(model, 'estimators_') else "Unknown",
            "n_estimators": len(getattr(model, 'estimators_', [])),
            "available": True
        }
    
    if BERT_AVAILABLE and 'BERT' in models:
        model_details['BERT Fine-tuning'] = {
            "type": "BERTMultiLabelClassifier",
            "base_model": "distilbert-base-uncased",
            "available": True,
            "fine_tuner_available": 'BERT_fine_tuner' in models
        }
    
    return jsonify({
        "models": model_details,
        "total_models": len(model_details),
        "label_count": len(label_columns)
    })

@app.route('/test_predict')
def test_predict():
    """Test tahmin endpoint'i"""
    test_text = request.args.get('text', 'Türkiye Merkez Bankası faiz oranlarını yükselterek enflasyonla mücadele etmeye devam ediyor.')
    
    try:
        results = predict_with_models(test_text)
        return jsonify({
            "status": "success",
            "test_text": test_text,
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "test_text": test_text
        }), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Upload klasörünü oluştur
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Modelleri yükle
    if not load_models():
        print("💥 Modeller yüklenemedi! Uygulama başlatılamıyor.")
        exit(1)
    
    print("🚀 Flask uygulaması başlatılıyor...")
    print("📍 Uygulama adresi: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
