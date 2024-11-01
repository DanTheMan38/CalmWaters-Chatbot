# CalmWaters-Chatbot

An NLP (DistilBERT) chatbot designed to assist guests with booking inquiries, FAQs, and general information for our vacation rental properties. This project integrates a Python backend with HTML for easy website integration. The bot provides quick responses about availability, property details, and booking options.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contact](#contact)
- [License](#license)

## Features

- **Natural Language Processing:** Utilizes DistilBERT for understanding and processing user queries.
- **Booking Assistance:** Provides quick responses about availability, property details, and booking options.
- **FAQs Handling:** Answers frequently asked questions to assist guests efficiently.
- **Easy Integration:** Combines a Python backend with HTML, CSS, and JavaScript for seamless website integration.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/CalmWaters-Chatbot.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd CalmWaters-Chatbot
   ```

3. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install flask transformers pandas scikit-learn markdown
   ```

   *Note:* Ensure you have Python 3.x installed on your system.

5. **Set Up the Fine-Tuned Model**

   - Place your fine-tuned DistilBERT model in the `fine_tuned_model` directory.
   - Ensure the paths in `app.py` and `train.py` point to the correct locations.

## Usage

1. **Run the Flask Application**

   ```bash
   python app.py
   ```

2. **Access the Chatbot Interface**

   Open your web browser and navigate to:

   ```
   http://localhost:5000/
   ```

3. **Interact with the Chatbot**

   - Type your questions into the input field.
   - The chatbot will provide responses based on the trained data.

## Project Structure

- **`app.py`**

  The main Flask application that handles routing, user input, and generating responses using the fine-tuned DistilBERT model.

- **`train.py`**

  Script used to train and fine-tune the DistilBERT model with custom intent data from `faq_data.csv`.

- **`data/faq_data.csv`**

  A CSV file containing intents, questions, and answers used for training the model.

- **`templates/chat.html`**

  HTML template for the chatbot interface.

- **`static/css/styles.css`**

  Contains the styling for the chatbot interface.

- **`static/js/script.js`**

  JavaScript code handling frontend interactions and animations.

- **`fine_tuned_model/`**

  Directory where the fine-tuned DistilBERT model is stored.

## Configuration

- **Model and Data Paths**

  Ensure that the file paths in `app.py` and `train.py` point to the correct locations of your model and data files.

  ```python
  # Example from app.py
  model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_model')
  tokenizer = DistilBertTokenizerFast.from_pretrained('fine_tuned_model')
  data_df = pd.read_csv('data/faq_data.csv')
  ```

- **Environment Variables**

  If you have any API keys or environment-specific settings, configure them as needed.

## Testing

- **Run Test Inputs**

  Use the provided `test_inputs` in `train.py` to evaluate the model's responses.

- **Sample Questions**

  - "What services does Calm Waters Plett offer?"
  - "How can I book a stay with Calm Waters Plett?"
  - "Are there any special offers or promotions currently available?"

- **Evaluating Responses**

  Ensure the chatbot provides accurate and helpful responses to user queries.

## Contact

Daniel Barry

- **Email:** [danielbarry36@gmail.com](mailto:danielbarry36@gmail.com)
- **GitHub:** [danielbarry36](https://github.com/danielbarry36)

Feel free to reach out via email or GitHub for any questions or contributions.

## License

This project does not have an assigned license.

---

*Note:* Integration with a live server is not included, as the client has opted out of server deployment due to associated costs and project complexity.