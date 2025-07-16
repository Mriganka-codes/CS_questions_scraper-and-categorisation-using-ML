A Python-based tool for educators and students that scrapes computer‑science questions from legit sources from the internet, preprocesses them, and automatically classifies them into categories using state of the art BERT technique and a self developed heuristic method.
**Tech Stack**
Frontend: HTML + Flask Templates
Backend: Python (Flask)
ML Frameworks: PyTorch, HuggingFace Transformers
PDF Export: fpdf, reportlab
Data Handling: Pandas, NumPy, CSV, Regex

**How It Works**
Select a Topic
Choose from 8 CS topics — questions are scraped in real-time.

**Choose Classification Method**
Upload scraped questions and select one of:
Rule-Based Classifier
BERT Model (pre-trained)
Generate Results
Download CSVs with predicted difficulty levels or auto-generate a question paper PDF.

**BERT Model Performance (Validation Accuracy)**
Average Accuracy: ~68.8%
Best Epoch Accuracy: 75%
F1 Macro Score: ~0.75

