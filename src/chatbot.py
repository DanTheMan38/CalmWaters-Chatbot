from flask import Flask, render_template, request, jsonify
import markdown
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('C:/Users/Daniel/Documents/calmwaters-chatbot/fine_tuned_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('C:/Users/Daniel/Documents/calmwaters-chatbot/fine_tuned_model')

# Create a pipeline for text classification
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Load your data into a DataFrame
data_df = pd.read_csv('C:/Users/Daniel/Documents/calmwaters-chatbot/data/faq_data.csv')

# Create a mapping from model labels to intent names
intent_mapping = {
    'LABEL_0': 'Home',
    'LABEL_1': 'LocalArea',
    'LABEL_2': 'AboutUs',
    'LABEL_3': 'Properties',
    'LABEL_4': 'Contact',
}

# Function to ensure each sentence ends with a full stop
def ensure_full_stop(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Check if the sentence ends with a period, exclamation mark, or question mark
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        processed_sentences.append(sentence)
    # Join the sentences back into a single string
    return ' '.join(processed_sentences)

def get_response(user_input):
    # Use the classifier to predict the intent
    prediction = classifier(user_input)[0]
    label = prediction['label']
    score = prediction['score']

    print(f"Predicted label: {label}, Intent: {intent_mapping.get(label)}, Score: {score}")

    # Map the model's label to the actual intent
    intent = intent_mapping.get(label)

    # Check if the intent mapping exists
    if intent is None:
        return "I'm sorry, I didn't quite understand that. Could you please rephrase? If you need further assistance, feel free to reach out through our contact section."

    # Threshold to ensure confidence in prediction
    if score < 0.5:
        return "I'm sorry, I didn't quite understand that. Could you please rephrase? If you need further assistance, feel free to reach out through our contact section."

    # Filter the data for the identified intent
    intent_data = data_df[data_df['Intent'] == intent]

    if intent_data.empty:
        return "I'm sorry, I couldn't find an answer to your question. Please feel free to reach out through our contact section, and we'll be happy to assist you."

    # Use TfidfVectorizer to vectorize the questions
    vectorizer = TfidfVectorizer()
    vectorizer.fit(intent_data['Question'])

    # Vectorize the user's question
    user_question_vec = vectorizer.transform([user_input])

    # Vectorize the questions in the data
    data_question_vecs = vectorizer.transform(intent_data['Question'])

    # Compute cosine similarity
    similarities = cosine_similarity(user_question_vec, data_question_vecs)

    # Get the index of the most similar question
    most_similar_idx = similarities.argmax()

    # Get the most similar question and its similarity score
    most_similar_score = similarities[0, most_similar_idx]

    # Threshold for similarity
    if most_similar_score < 0.3:  # Adjust threshold as needed
        return "I'm sorry, I couldn't find an answer to your question. Please feel free to reach out through our contact section, and we'll be happy to assist you."

    # Get the corresponding answer
    answer = intent_data.iloc[most_similar_idx]['Answer']

    answer = ensure_full_stop(answer)

    return answer

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'answer': "Invalid request. Please provide a question."}), 400

    user_question = data['question']
    answer = get_response(user_question)

    # Convert the answer from Markdown to HTML
    answer_html = markdown.markdown(answer)

    return jsonify({'answer': answer_html})

if __name__ == "__main__":
    app.run(debug=True)