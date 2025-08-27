# 🤖 Multi-Label Text Classification Web Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![BERT](https://img.shields.io/badge/BERT-DistilBERT-red.svg)](https://huggingface.co/distilbert-base-uncased)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive Flask web application that automatically categorizes texts using multiple trained machine learning models, including traditional ML algorithms and modern BERT-based transformers.

## 🌟 Key Features

### 🧠 Advanced ML Models
- **Logistic Regression** with MultiOutput strategy
- **Random Forest** with ensemble learning
- **Support Vector Machine (SVM)** with optimal hyperparameters
- **Classifier Chain** models for label dependency modeling
- **Probabilistic Classifier Chain** with uncertainty quantification
- **BERT Fine-tuning** (DistilBERT) with correlation-aware loss functions

### 🏷️ Multi-Label Classification Categories
- **Government & Social Affairs** - Political news, social policies
- **Economics & Finance** - Market analysis, financial reports  
- **Markets & Trade** - Commerce, business transactions
- **Business & Industry** - Corporate news, industrial developments

### ✨ Web Interface Features
- **Modern Responsive Design** - Works on all devices
- **Real-time Character Counter** - Input validation and feedback
- **Sample Texts** - Quick testing with pre-loaded examples
- **Comparative Results** - Side-by-side model performance
- **Confidence Score Visualization** - Interactive charts and graphs
- **Export Capabilities** - Download results in JSON/CSV formats
- **RESTful API** - Programmatic access for integration

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip package manager
16GB+ RAM (recommended for BERT models)
```

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/multi-label-text-classifier.git
cd multi-label-text-classifier
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Pre-trained Models**
```bash
# Models are included in the repository under saved_models/
# Ensure the following structure exists:
saved_models/
└── session_2_20250825_062911/
    ├── tfidf_vectorizer.pkl
    ├── multi_logistic_regression.pkl
    ├── multi_random_forest.pkl
    ├── classifier_chain_lr.pkl
    ├── probabilistic_chain.pkl
    ├── bert_finetuned_model.pkl
    └── ...
```

5. **Launch Application**
```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## 🔧 Usage

### Web Interface
1. Navigate to the main page
2. Enter your text in the input field (up to 10,000 characters)
3. Click "Classify" to get predictions from all models
4. Compare results across different algorithms
5. Download results if needed

### API Usage

#### Predict Endpoint
```python
import requests

# POST request to prediction endpoint
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Your text to classify here'})

results = response.json()
print(results['predictions'])
```

#### Health Check
```python
# Check application status
health = requests.get('http://localhost:5000/health')
print(health.json())
```

#### API Response Format
```json
{
  "original_text": "Input text",
  "processed_text": "Cleaned and processed text",
  "predictions": {
    "Logistic Regression": {
      "predicted_labels": ["Economics & Finance"],
      "label_details": [
        {
          "label": "Economics & Finance",
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
    "consensus_labels": ["Economics & Finance"],
    "most_predicted_labels": [["Economics & Finance", 4]]
  }
}
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Application                     │
├─────────────────────────────────────────────────────────────┤
│  Web Interface  │  RESTful API  │  Model Management        │
├─────────────────────────────────────────────────────────────┤
│                  Text Preprocessing Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│  TF-IDF         │  BERT          │  Feature Engineering     │
│  Vectorization  │  Tokenization  │  & Selection             │
├─────────────────────────────────────────────────────────────┤
│                    Model Ensemble                           │
├─────────────────────────────────────────────────────────────┤
│ Logistic │ Random │   SVM   │ Classifier │ Probabilistic │ BERT │
│Regression│ Forest │         │   Chain    │    Chain      │      │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture Details

#### Traditional ML Pipeline
- **Feature Extraction**: TF-IDF with n-grams (1,2), 50K max features
- **Preprocessing**: Text cleaning, normalization, stop-word removal
- **Multi-label Strategy**: OneVsRest and MultiOutput classifiers
- **Label Dependencies**: Classifier Chain with correlation-based ordering

#### BERT Integration
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning**: Custom multi-label head with dropout regularization
- **Loss Functions**: Binary Cross-Entropy + Focal Loss for imbalanced data
- **Optimization**: AdamW optimizer with linear warmup scheduling

## 📊 Model Performance

| Model | F1-Score (Macro) | Accuracy | Training Time | Inference Time |
|-------|------------------|----------|---------------|----------------|
| **Logistic Regression** | 0.925 | 0.840 | 5.8s | ~15ms |
| **Probabilistic Chain** | 0.915 | 0.841 | 390.8s | ~45ms |
| **Random Forest** | 0.890 | 0.783 | 17.5s | ~25ms |
| **Classifier Chain (LR)** | 0.914 | 0.840 | 6.4s | ~20ms |
| **Classifier Chain (RF)** | 0.865 | 0.745 | 17.1s | ~30ms |
| **BERT Fine-tuned** | 0.738 | 0.664 | 747.8s | ~200ms |

### Performance Insights
- **Logistic Regression** offers the best balance of speed and accuracy
- **Probabilistic Chain** provides uncertainty quantification
- **BERT** excels at capturing semantic relationships but requires more resources
- **Ensemble approaches** can combine strengths of multiple models

## 🔬 Technical Details

### Text Preprocessing Pipeline
1. **Normalization**: Convert to lowercase
2. **Cleaning**: Remove special characters, keep only alphanumeric + spaces
3. **Whitespace**: Normalize multiple spaces to single space
4. **Validation**: Filter empty texts and validate length

### Feature Engineering
- **TF-IDF Parameters**: 
  - Max features: 50,000
  - N-gram range: (1, 2)
  - Min document frequency: 2
  - Max document frequency: 0.95
- **Correlation Analysis**: Label dependency modeling
- **Feature Selection**: Automatic relevance determination

### Model Training Strategy
- **Cross-validation**: Stratified K-fold for robust evaluation
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Early Stopping**: Prevent overfitting in neural models
- **Model Persistence**: Automatic saving with versioning

## 📁 Project Structure

```
multi-label-text-classifier/
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── templates/                         # HTML templates
│   ├── base.html                     # Base template
│   ├── index.html                    # Main page
│   ├── results.html                  # Results display
│   ├── 404.html                      # Error pages
│   └── 500.html
├── static/                           # Static assets
│   ├── css/
│   │   └── style.css                # Custom styles
│   └── js/
│       └── main.js                  # JavaScript functions
├── saved_models/                     # Pre-trained models
│   └── session_2_20250825_062911/
│       ├── *.pkl                    # Serialized models
│       └── *.json                   # Configuration files
├── notebooks/                        # Jupyter notebooks
│   ├── dataset-Analysis-and-preprocessing.ipynb
│   └── multi_model_classifier.ipynb
├── test_api.py                       # API testing script
└── uploads/                          # File upload directory
```

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/multi-label-text-classifier.git
cd multi-label-text-classifier

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Start development server with hot reload
export FLASK_ENV=development
python app.py
```

### Adding New Models

1. **Implement Model Class**: Follow scikit-learn estimator interface
2. **Update Model Loading**: Add to `load_models()` function
3. **Modify Prediction Pipeline**: Include in `predict_with_models()`
4. **Update Templates**: Add to web interface display

### Custom Loss Functions

The system includes several advanced loss functions for multi-label classification:

```python
# Correlation-aware loss
class CorrelationAwareLoss:
    @staticmethod
    def pairwise_ranking_loss(y_true, y_pred, margin=1.0):
        # Implementation for label correlation modeling
        pass
    
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        # Implementation for handling class imbalance
        pass
```

## 🔒 Security Features

- **Input Validation**: Text length limits and content sanitization
- **File Upload Security**: Size limits (16MB) and type restrictions
- **CSRF Protection**: Built-in Flask security measures
- **Error Handling**: Comprehensive exception management
- **Rate Limiting**: Configurable request throttling

## 🚨 Important Notes

### File Path Configuration
When deploying to different environments, update the following paths:

**In `app.py`:**
```python
MODEL_DIR = 'path/to/your/saved_models/session_XXXXXX_XXXXXX/'
```

**In Jupyter notebooks:**
- Dataset paths: `preprocessed_dataset_3.csv`
- Output directories for models and visualizations
- Model loading paths

### System Requirements
- **Memory**: 8GB RAM minimum, 16GB recommended for BERT models
- **Storage**: 2GB for models and dependencies
- **CPU**: Multi-core processor recommended for ensemble models

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Errors**
   - Verify model files exist in `saved_models/` directory
   - Check file permissions and paths
   - Ensure numpy version compatibility (1.26.4+ recommended)

3. **BERT Memory Issues**
   ```bash
   # Reduce batch size or disable BERT if insufficient memory
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

4. **Port Conflicts**
   ```python
   # Change port in app.py
   app.run(host='0.0.0.0', port=8080, debug=False)
   ```

5. **Numpy Version Conflicts**
   ```bash
   pip install numpy==1.26.4 pandas==2.2.0 scikit-learn==1.5.0 --upgrade
   ```

## 📈 Performance Optimization

### Production Deployment
- Use **Gunicorn** or **uWSGI** for production WSGI server
- Implement **Redis** caching for model predictions
- Set up **nginx** reverse proxy for static file serving
- Configure **logging** and monitoring

### Scaling Considerations
- **Horizontal Scaling**: Deploy multiple app instances
- **Model Serving**: Separate model inference service
- **Database**: Store results and user interactions
- **Load Balancing**: Distribute traffic across instances

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📊 Datasets

The project uses preprocessed text classification datasets with the following characteristics:

- **Format**: CSV with text and multi-label columns
- **Size**: ~158K samples with 49 possible labels
- **Languages**: Primarily English text
- **Domains**: News articles, government documents, financial reports

### Data Preprocessing
- Text normalization and cleaning
- Label encoding and correlation analysis
- Train/validation/test splits (70/15/15)
- TF-IDF vectorization with optimized parameters

## 🎯 Use Cases

### Business Applications
- **Content Moderation**: Automatically categorize user-generated content
- **Document Classification**: Organize large document repositories
- **News Categorization**: Real-time news article classification
- **Email Filtering**: Multi-label email classification and routing

### Research Applications
- **Comparative ML Studies**: Benchmark different algorithms
- **Label Dependency Research**: Study multi-label relationships
- **Transfer Learning**: Fine-tune models for specific domains
- **Ensemble Methods**: Explore model combination strategies

## 📚 References & Citations

If you use this project in your research, please cite:

```bibtex
@software{multi_label_text_classifier,
  title={Multi-Label Text Classification Web Application},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multi-label-text-classifier}
}
```

### Related Work
- [Multi-label Classification: A Survey](https://link.springer.com/article/10.1007/s10994-013-5413-3)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Classifier Chains for Multi-label Classification](https://link.springer.com/chapter/10.1007/978-3-642-04174-7_17)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/multi-label-text-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multi-label-text-classifier/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML library
- **Hugging Face** for transformer models and tokenizers
- **Flask** community for the web framework
- **Contributors** who helped improve this project

---

**⚠️ Note**: This application is developed for educational and research purposes. For production use, additional security testing and optimization are recommended.

**🌟 Star this repository** if you find it useful!
