from flask import Flask, Response, render_template, request, send_file, redirect, url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import io
import csv
from openpyxl import Workbook
import torch
import transformers
import numpy as np
from transformers import BertTokenizer
from sklearn import preprocessing

app = Flask(__name__)

# Load the model class definition
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Load the model and tokenizer once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTClass()
model.load_state_dict(torch.load('best_model.bin', map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label encoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.array(['easy', 'medium', 'hard'])

def preprocess_questions(questions, tokenizer, max_len=200):
    inputs = tokenizer.batch_encode_plus(
        questions,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
        return_attention_mask=True,
        truncation=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
    return ids, mask, token_type_ids

def predict_difficulties(questions, model, tokenizer, label_encoder, batch_size=8):
    model.eval()
    ids, mask, token_type_ids = preprocess_questions(questions, tokenizer)
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    all_predictions = []
    for i in range(0, len(questions), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_mask = mask[i:i+batch_size]
        batch_token_type_ids = token_type_ids[i:i+batch_size]

        with torch.no_grad():
            outputs = model(batch_ids, batch_mask, batch_token_type_ids)
            probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)
            all_predictions.extend(predicted_classes)

    return label_encoder.inverse_transform(all_predictions)
def extract_features(question):
    options_index = question.find(" Options:")
    question_text = question[:options_index]
    features = {}
    features['length'] = len(question_text.split())  # Calculate length
    features['has_numericals'] = any(char.isdigit() for char in question_text)  # Check presence of numericals
    keywords = ['calculate', 'match', 'predict', '*', '|', 'grammar', '->', 'bits', 'error']  # Define keywords
    features['has_keywords'] = any(keyword in question_text.lower() for keyword in keywords)  # Check presence of keywords
    return features

def categorize_question(features):
    if features['length'] <= 40 and not features['has_numericals'] and not features['has_keywords']:
        return 'easy'
    if features['length'] <= 50 and features['has_numericals']:
        return 'medium'
    if features['length'] <= 50 and features['has_keywords']:
        return 'medium'
    else:
        return 'hard'

def predict_difficulties_model_2(questions):
    predictions = []
    for question in questions:
        features = extract_features(question)
        category = categorize_question(features)
        predictions.append(category)
    return predictions


def clean_text(text):
    """Remove unwanted characters and clean the text, including serial numbers."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'^\d+[\.\)]\s+', '', text)  # Remove serial numbers at the beginning
    return text.strip()

def scrape_website_1(url):
    questions_data = []
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        questions = soup.find_all("p", class_="pq")
        options = soup.find_all("ol", class_="pointsa")
        
        for question, option in zip(questions, options):
            # Skip questions that have a code block
            if question.find_next("div", class_="codeblock"):
                continue
            
            question_text = clean_text(question.get_text())
            
            option_texts = [clean_text(li.get_text()) for li in option.find_all("li")]
            combined_question = question_text + " Options: " + "; ".join(option_texts)
            
            questions_data.append({
                "Question": combined_question
            })
    else:
        print(f"Failed to retrieve the webpage: {url}")
    
    return questions_data

def scrape_website_2(url):
    questions_data = []
    page = 1
    while True:
        page_url = f"{url}?page={page}"
        response = requests.get(page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            question_containers = soup.find_all('div', class_='QuizQuestionCard_quizCard__quizQuestionTextContainer__CzNWr')
            if question_containers:
                for question_container in question_containers:
                    question_text = clean_text(question_container.get_text(strip=True))
                    options_container = question_container.find_next_sibling('ul', class_='QuizQuestionCard_quizCard__optionsList__4pILJ')
                    if options_container:
                        option_texts = [clean_text(li.get_text()) for li in options_container.find_all('li')]
                        combined_question = question_text + " Options: " + "; ".join(option_texts)
                        questions_data.append({"Question": combined_question})
                page += 1
            else:
                break  # No more questions found, exit loop
        else:
            print(f"Failed to retrieve the webpage: {page_url}")
            break  # Exit loop on error
    return questions_data
    # ... (existing scrape_website_2 function)

def scrape_website_3(url):
    questions_data = []
    page_number = 1
    
    while True:
        page_url = f"{url}/{page_number}" if page_number > 1 else url
        response = requests.get(page_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            questions = soup.find_all("div", class_="w-100")
            
            for question_div in questions:
                question_text = clean_text(question_div.get_text())
                option_container = question_div.find_next("div", class_="option-container")
                if option_container:
                    option_texts = [clean_text(option.get_text()) for option in option_container.find_all("div")]
                    combined_question = question_text + " Options: " + "; ".join(option_texts)
                    
                    questions_data.append({
                        "Question": combined_question
                    })
            
            page_number += 1
        else:
            print(f"No more pages or failed to retrieve the webpage: {page_url}")
            break
    
    return questions_data
  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    topic = request.form.get('topic')

    if topic == 'Operating System':
        urls = [
            "https://www.geeksforgeeks.org/quizzes/operating-systems-gq/process-synchronization-gq/",
            "https://www.javatpoint.com/operating-system-mcq",
            "https://compsciedu.com/mcq-questions/Operating-System/NET-computer-science-question-paper",
            "https://www.geeksforgeeks.org/quizzes/operating-systems-gq/memory-management-gq/"
        ]
    elif topic == 'Programming':
        urls = [
            "https://www.geeksforgeeks.org/quizzes/c-quiz-101-gq/",
            "https://www.geeksforgeeks.org/quizzes/c-plus-plus-gq/class-and-object-gq/",
            "https://www.geeksforgeeks.org/quizzes/functions-python-gq/",

        ]
    elif topic == 'DBMS':
        urls = [
            "https://compsciedu.com/mcq-questions/DBMS/NET-computer-science-question-paper",
            "https://www.javatpoint.com/dbms-mcq",
            "https://www.javatpoint.com/data-mining-mcq",
            "https://www.geeksforgeeks.org/quizzes/dbms-gq/er-and-relational-models-gq/",
            "https://www.geeksforgeeks.org/quizzes/dbms-gq/transactions-and-concurrency-control-gq/"
            
        ] 
    elif topic == 'Theory of Computation':
        urls = [
            "https://compsciedu.com/mcq-questions/Theory-of-Computation(TOC)/NET-computer-science-question-paper",
            "https://www.geeksforgeeks.org/quizzes/lexical-analysis-gq/",
            "https://www.geeksforgeeks.org/quizzes/code-generation-and-optimization-gq/",
            "https://www.javatpoint.com/compiler-design-mcq",

        ]
    elif topic == 'Software Engineering':
        urls = [
            "https://www.geeksforgeeks.org/quizzes/software-engineering-gq/",
            "https://www.javatpoint.com/software-engineering-mcq",
            "https://compsciedu.com/mcq-questions/Software-Engineering/NET-computer-science-question-paper",
        ]        
    elif topic == 'Algorithms':
        urls = [
            "https://www.geeksforgeeks.org/quizzes/algorithms-gq/top-mcqs-on-dynamic-programming-with-answers/",
            "https://www.geeksforgeeks.org/quizzes/algorithms-gq/top-mcqs-on-recursion-algorithm-with-answers/",
            "https://www.javatpoint.com/data-structure-mcq",
            "https://compsciedu.com/mcq-questions/Data-Structures-and-Algorithms/NET-computer-science-question-paper",
        ]   
    elif topic == 'Computer Networks':
        urls = [
            "https://www.javatpoint.com/computer-network-mcq",
            "https://www.geeksforgeeks.org/quizzes/misc-topics-in-computer-networks-gq/",
            "https://www.geeksforgeeks.org/quizzes/network-security-gq/",
            "https://www.geeksforgeeks.org/quizzes/ip-addressing-57/",
            "https://compsciedu.com/mcq-questions/Networking/NET-computer-science-question-paper",
        ] 
    elif topic == 'Discreet Maths':
        urls = [
            "https://www.javatpoint.com/discrete-mathematics-mcq",
            "https://compsciedu.com/mcq-questions/Discrete-Mathematics/Relations",
            "https://compsciedu.com/mcq-questions/Discrete-Mathematics/Sets-and-Functions"
        ]
    # Add more topic cases here

    questions_data = []
    for url in urls:
        if 'geeksforgeeks.org' in url:
            questions_data.extend(scrape_website_2(url))
        elif 'javatpoint.com' in url:
            questions_data.extend(scrape_website_1(url))
        elif 'compsciedu.com' in url:
            questions_data.extend(scrape_website_3(url))

    if questions_data:
        # Create a CSV file
        csv_output = io.StringIO()
        writer = csv.DictWriter(csv_output, fieldnames=['Question'])
        writer.writeheader()
        for question in questions_data:
            writer.writerow({'Question': question['Question']})
        csv_output.seek(0)

        return Response(
            csv_output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{topic}_questions.csv"'
            }
        )
    else:
        return "No questions found."


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        model_choice = request.form.get('model_choice')
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if 'Question' in df.columns:
                questions = df['Question'].tolist()
                if model_choice == 'model_1':
                    difficulties = predict_difficulties(questions, model, tokenizer, label_encoder, batch_size=8)
                elif model_choice == 'model_2':
                    difficulties = predict_difficulties_model_2(questions)
                else:
                    return "Invalid model choice."

                df['Difficulty'] = difficulties

                # Create a new CSV file with the predictions
                output = io.StringIO()
                df.to_csv(output, index=False)
                output.seek(0)

                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    as_attachment=True,
                    download_name='predicted_difficulties.csv',
                    mimetype='text/csv'
                )
            else:
                return "CSV file must contain a 'Question' column."
        else:
            return "Invalid file format. Please upload a CSV file."
    return render_template('predict.html')



if __name__ == '__main__':
    app.run(debug=True)