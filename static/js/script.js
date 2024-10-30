const chatbotIcon = document.getElementById('chatbot-icon');
const chatWindow = document.getElementById('chat-window');
const closeBtn = document.getElementById('close-btn');

// Existing JavaScript code for handling messages
const form = document.getElementById('chat-form');
const messages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button'); // Get the send button

// Disable the send button initially
sendButton.disabled = true;

// Add event listener to input field to enable send button
userInput.addEventListener('input', function() {
    if (userInput.value.trim() !== '') {
        sendButton.disabled = false;
        sendButton.classList.add('active');
    } else {
        sendButton.disabled = true;
        sendButton.classList.remove('active');
    }
});

// Handle click on chatbot icon
chatbotIcon.addEventListener('click', () => {
    // Hide the chatbot icon immediately
    chatbotIcon.classList.add('hidden');

    // Ensure the chat window is visible
    chatWindow.style.display = 'block';

    // Force reflow
    void chatWindow.offsetWidth;

    // Add the 'open' class to trigger the opening animation
    chatWindow.classList.add('open');
});

// Handle click on close button
closeBtn.addEventListener('click', () => {
    // Start the close animation
    chatWindow.classList.remove('open');

    // Wait for the animation to finish
    setTimeout(() => {
        chatWindow.style.display = 'none';

        // Show the chatbot icon with slide-up animation
        chatbotIcon.classList.remove('hidden');
        chatbotIcon.classList.add('showing');

        // Remove the 'showing' class after animation completes
        setTimeout(() => {
            chatbotIcon.classList.remove('showing');
        }, 300); // Duration matches the slideUp animation
    }, 300); // Duration matches the transform transition
});

// Reset the animation class
function removeAnimation() {
    chatbotIcon.classList.remove('showing');
    void chatbotIcon.offsetWidth; // Force a reflow to reset the animation
}

// Handle form submission
form.addEventListener('submit', function(e) {
    e.preventDefault();
    const question = userInput.value.trim();
    if (question === '') return;
    appendMessage('You', question, 'user');
    userInput.value = '';
    sendButton.disabled = true;
    sendButton.classList.remove('active');

    // Show typing indicator
    const typingIndicator = showTypingIndicator();

    fetch('/get_answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => {
            // Remove typing indicator
            messages.removeChild(typingIndicator);
            // Append the bot's message
            appendMessage('Chatbot', data.answer, 'bot', true);
        }, 2000); // 2-second delay
    })
    .catch(error => {
        messages.removeChild(typingIndicator);
        appendMessage('Chatbot', 'Sorry, something went wrong.', 'bot', false);
        console.error('Error:', error);
    });
});

function appendMessage(sender, text, className, isHTML = false) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', className);

    let messageElement;

    if (isHTML) {
        // Use a div for HTML content to avoid invalid nesting
        messageElement = document.createElement('div');
        messageElement.classList.add('message-content');
    } else {
        // Use a p element for plain text
        messageElement = document.createElement('p');
    }

    messageContainer.appendChild(messageElement);
    messages.appendChild(messageContainer);
    messages.scrollTop = messages.scrollHeight;

    if (className === 'bot') {
        // Start the typing effect
        typeWriterEffect(messageElement, text, isHTML);
    } else {
        // For user messages
        if (isHTML) {
            messageElement.innerHTML = text;
        } else {
            messageElement.textContent = text;
        }
    }
}

function typeWriterEffect(element, text, isHTML, speed = 30) {
    if (isHTML) {
        // Strip HTML tags for the typing effect
        const tempElement = document.createElement('div');
        tempElement.innerHTML = text;
        const plainText = tempElement.textContent || tempElement.innerText || '';
        let index = 0;

        function type() {
            if (index < plainText.length) {
                element.textContent += plainText.charAt(index);
                index++;
                setTimeout(type, speed);
            } else {
                // After typing is complete, set the innerHTML to render HTML content
                element.innerHTML = text;
            }
        }
        type();
    } else {
        let index = 0;
        function type() {
            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;
                setTimeout(type, speed);
            }
        }
        type();
    }
}

function showTypingIndicator() {
    const typingContainer = document.createElement('div');
    typingContainer.classList.add('message', 'bot', 'typing');

    const typingBubble = document.createElement('div');
    typingBubble.classList.add('typing-bubble');

    // Create three dots for the typing animation
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('typing-dot');
        typingBubble.appendChild(dot);
    }

    typingContainer.appendChild(typingBubble);
    messages.appendChild(typingContainer);
    messages.scrollTop = messages.scrollHeight;
    return typingContainer;
}