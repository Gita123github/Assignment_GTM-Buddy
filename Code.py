import pandas as pd
import json
import nltk
import torch
from flask import Flask, request, jsonify
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
import spacy

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Flask app
app = Flask(__name__)

# Data Preprocessing
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Load and preprocess dataset
data = pd.read_csv('calls_dataset.csv')
data['cleaned_text'] = data['text_snippet'].apply(preprocess_text)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Multi-label classification setup
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_data['labels'].str.split(', '))
test_labels = mlb.transform(test_data['labels'].str.split(', '))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class CallDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {**encoding, "labels": label}

train_dataset = CallDataset(train_data['cleaned_text'].tolist(), train_labels)
test_dataset = CallDataset(test_data['cleaned_text'].tolist(), test_labels)

# Train classification model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(mlb.classes_))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Save model
model.save_pretrained('./saved_model')

# Entity Extraction
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Load domain knowledge
with open('domain_knowledge.json', 'r') as f:
    domain_knowledge = json.load(f)

patterns = []
for category, items in domain_knowledge.items():
    for item in items:
        patterns.append({"label": category.upper(), "pattern": item})

ruler.add_patterns(patterns)

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Summarization
summarizer = pipeline("summarization", model="t5-small")

# Flask API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')

    # Classification
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    labels = torch.sigmoid(outputs.logits).detach().numpy()
    predicted_labels = [mlb.classes_[i] for i in range(len(labels[0])) if labels[0][i] > 0.5]

    # Entity Extraction
    entities = extract_entities(text)

    # Summarization
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']

    return jsonify({
        "predicted_labels": predicted_labels,
        "extracted_entities": entities,
        "summary": summary
    })

if __name__ == '__main__':
    app.run(debug=True)
