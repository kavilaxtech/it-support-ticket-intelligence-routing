# IT Support Ticket Intelligence & Automated Routing System

NLP prototype that analyzes IT support tickets, predicts category/issue/priority, routes tickets to a support team, retrieves similar historical tickets, and surfaces a resolution suggestion.

## Features
- Multi-task NLP classification: category, issue type, priority
- TF-IDF unigram/bigram feature extraction
- Logistic Regression classifiers
- Confidence estimates from model probabilities
- Category-to-team routing
- TF-IDF cosine-similarity historical ticket retrieval
- Resolution suggestion from similar tickets
- Streamlit dashboard
- Accuracy, precision, recall, F1 and confusion matrices

## Data
The included dataset is synthetic and created specifically for this project.

## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
link: https://it-support-ticket-intelligence-routing.streamlit.app/
