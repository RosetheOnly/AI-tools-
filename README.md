# AI Tools Assignment - Team AIgineers

## Project Overview
This repository contains implementations of:
1. Iris flower classification using Scikit-learn
2. MNIST digit recognition with TensorFlow CNN
3. Amazon review analysis with spaCy NLP
4. Streamlit deployment for MNIST classifier

## Team Members
- Sabulkong Valentine: Iris Classification
- Roda Nyamai: MNIST CNN Implementation
- Sabulkong Valentine: NLP Analysis
- Michael Randa: Deployment & Ethics

## Setup
```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run applications
python src/iris_classifier.py
python src/mnist_cnn.py
python src/nlp_analysis.py
streamlit run deployment/streamlit_app.py
