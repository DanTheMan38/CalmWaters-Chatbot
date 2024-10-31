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
    'LABEL_0': 'has_fire_extinguisher',
    'LABEL_1': 'has_first_aid',
    'LABEL_2': 'nearby_attractions',
    'LABEL_3': 'get_bathrooms',
    'LABEL_4': 'meet_the_team',
    'LABEL_5': 'checkin_checkout',
    'LABEL_6': 'local_activities',
    'LABEL_7': 'contact_management',
    'LABEL_8': 'about_us',
    'LABEL_9': 'restaurants',
    'LABEL_10': 'consumables_cost',
    'LABEL_11': 'has_washing_machine',
    'LABEL_12': 'has_heat',
    'LABEL_13': 'best_time_to_visit',
    'LABEL_14': 'diy_fee',
    'LABEL_15': 'has_coffee_tea',
    'LABEL_16': 'max_stay',
    'LABEL_17': 'goodbye',
    'LABEL_18': 'weekly_retainer',
    'LABEL_19': 'booking_inquiry',
    'LABEL_20': 'has_tumble_dryer',
    'LABEL_21': 'guest_experience',
    'LABEL_22': 'personal_booking_commission',
    'LABEL_23': 'has_dishwasher',
    'LABEL_24': 'pet_policy',
    'LABEL_25': 'contact_email',
    'LABEL_26': 'payment_methods',
    'LABEL_27': 'staging_fee',
    'LABEL_28': 'price_high',
    'LABEL_29': 'has_charcoal_wood',
    'LABEL_30': 'explore_properties',
    'LABEL_31': 'retainer_adjustment',
    'LABEL_32': 'has_iron',
    'LABEL_33': 'has_microwave',
    'LABEL_34': 'get_type',
    'LABEL_35': 'local_commitment',
    'LABEL_36': 'special_requests',
    'LABEL_37': 'transportation',
    'LABEL_38': 'min_stay',
    'LABEL_39': 'get_place',
    'LABEL_40': 'property_amenities',
    'LABEL_41': 'total_property_count',
    'LABEL_42': 'cabin_count',
    'LABEL_43': 'plettenberg_bay_overview',
    'LABEL_44': 'price_mid',
    'LABEL_45': 'get_address',
    'LABEL_46': 'pricing_inquiry',
    'LABEL_47': 'allows_pets',
    'LABEL_48': 'has_biscuits',
    'LABEL_49': 'apartment_count',
    'LABEL_50': 'top_attractions',
    'LABEL_51': 'cancellation_policy',
    'LABEL_52': 'how_are_you',
    'LABEL_53': 'has_desk',
    'LABEL_54': 'get_pax',
    'LABEL_55': 'housekeeping_services',
    'LABEL_56': 'has_fan',
    'LABEL_57': 'get_size',
    'LABEL_58': 'property_name',
    'LABEL_59': 'contact_address',
    'LABEL_60': 'mission_vision',
    'LABEL_61': 'checkin_instructions',
    'LABEL_62': 'getting_to_plettenberg_bay',
    'LABEL_63': 'emergency_contact',
    'LABEL_64': 'has_rusk',
    'LABEL_65': 'greeting',
    'LABEL_66': 'early_late_checkout',
    'LABEL_67': 'has_bbq',
    'LABEL_68': 'website_name',
    'LABEL_69': 'commission_rate',
    'LABEL_70': 'checkin_method',
    'LABEL_71': 'services_included',
    'LABEL_72': 'bi_monthly_retainer',
    'LABEL_73': 'property_management_services',
    'LABEL_74': 'has_tv',
    'LABEL_75': 'maintenance_fee',
    'LABEL_76': 'has_milk',
    'LABEL_77': 'group_booking',
    'LABEL_78': 'office_hours',
    'LABEL_79': 'business_info',
    'LABEL_80': 'lodge_count',
    'LABEL_81': 'get_bed_sizes',
    'LABEL_82': 'website_services',
    'LABEL_83': 'contact_whatsapp',
    'LABEL_84': 'get_bedrooms',
    'LABEL_85': 'nightlife',
    'LABEL_86': 'property_availability',
    'LABEL_87': 'house_count',
    'LABEL_88': 'get_built_year',
    'LABEL_89': 'cleaning_fee',
    'LABEL_90': 'has_hair_dryer',
    'LABEL_91': 'has_water',
    'LABEL_92': 'response_time',
    'LABEL_93': 'guesthouse_room_count',
    'LABEL_94': 'plettenberg_bay_history',
    'LABEL_95': 'has_fireplace',
    'LABEL_96': 'property_views',
    'LABEL_97': 'contact_phone',
    'LABEL_98': 'price_low',
    'LABEL_99': 'has_body_wash',
    'LABEL_100': 'has_ac'
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