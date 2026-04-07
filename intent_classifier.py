"""
Neural Network Intent Classifier for Agentic AI
Uses scikit-learn MLP (Multi-layer Perceptron) to classify user intent
"""

import pickle
import os
from typing import List, Dict
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class IntentClassifier:
    """Neural Network Intent Classifier using MLP"""
    
    def __init__(self):
        self.model_path = "intent_model.pkl"
        self.pipeline = None
        self.intents = [
            'greeting', 'farewell', 'how_are_you', 'name_query',
            'help_request', 'sad_emotion', 'happy_emotion', 'joke_request',
            'gratitude', 'time_query', 'weather_query', 'affection',
            'general_chat'
        ]
        self.responses = {
            'greeting': [
                "Hey there! It's so nice to hear your voice. How's your day going?",
                "Hi! I can hear you loud and clear. What's up?",
                "Hello! I've been waiting for someone to talk to. What's on your mind?",
                "Hey! Ready for a great conversation?",
            ],
            'farewell': [
                "Take care! It's been wonderful talking with you. Come back anytime!",
                "Goodbye for now! I'll be here whenever you want to chat again. Have a great day!",
                "See you later! Thanks for the conversation!",
            ],
            'how_are_you': [
                "I'm doing great, thanks for asking! I'm an AI, so I don't have feelings exactly, but I'm genuinely excited to chat with you!",
                "Oh, you know, just processing information and looking forward to interesting conversations like this one! How about you?",
            ],
            'name_query': [
                "I'm Agentic AI, your voice companion. I love having natural conversations and learning about what you're thinking. What's your name?",
                "I'm Agentic AI! Think of me as your friendly digital assistant who's always ready to chat.",
            ],
            'help_request': [
                "I'd absolutely love to help! What specifically do you need assistance with?",
                "Sure thing! Tell me more about what you're trying to accomplish, and I'll do my best to guide you.",
                "Of course! I enjoy problem-solving. What's the situation?",
            ],
            'sad_emotion': [
                "I'm really sorry you're feeling that way. It sounds difficult, and I want you to know that your feelings matter. Do you want to talk about what's going on?",
                "That sounds really tough. Sometimes just saying it out loud can help a little bit. I'm here to listen if you need to talk.",
            ],
            'happy_emotion': [
                "That's wonderful! I love hearing good news! Tell me all about it!",
                "Yes! That's so exciting! What happened? Give me the details!",
                "Oh wow, that's fantastic! You must be feeling great right now!",
            ],
            'joke_request': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "I told my computer I needed a break, and now it won't stop sending me Kit-Kats!",
            ],
            'gratitude': [
                "You're so welcome! It's my pleasure to chat with you.",
                "No problem at all! I enjoy helping out.",
                "Anytime! That's what I'm here for. Thanks for talking with me!",
            ],
            'time_query': [
                "I can tell you the current time. Just a moment...",
            ],
            'weather_query': [
                "I can't actually check the weather since I'm an AI, but I hope it's nice wherever you are! Do you have plans to go outside?",
                "You know, I can't see outside, but I hope the weather is treating you well!",
            ],
            'affection': [
                "That's so sweet of you to say! I really enjoy our conversations too.",
                "Aww, you're making me blush! Well, if I could blush. But seriously, I appreciate you too!",
            ],
            'general_chat': [
                "That's really interesting! Tell me more about that.",
                "Hmm, I see what you mean. What else is on your mind?",
                "Oh, absolutely. This makes me curious - what's your take on where this is headed?",
                "Yeah, that makes a lot of sense. I wonder if there's more to explore here.",
                "Interesting! I haven't thought about it quite that way before. What led you to that conclusion?",
                "I hear you. Sometimes it's good to just talk things through, you know?",
            ]
        }
        
        # Training data: (text, intent)
        self.training_data = [
            # Greetings
            ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
            ("good morning", "greeting"), ("good afternoon", "greeting"), ("good evening", "greeting"),
            ("what's up", "greeting"), ("yo", "greeting"), ("greetings", "greeting"),
            ("howdy", "greeting"), ("hi there", "greeting"), ("hey there", "greeting"),
            
            # Farewells
            ("bye", "farewell"), ("goodbye", "farewell"), ("see you", "farewell"),
            ("see ya", "farewell"), ("later", "farewell"), ("talk to you later", "farewell"),
            ("ttyl", "farewell"), ("cya", "farewell"), ("have a good day", "farewell"),
            ("take care", "farewell"), ("catch you later", "farewell"),
            
            # How are you
            ("how are you", "how_are_you"), ("how're you", "how_are_you"),
            ("how you doing", "how_are_you"), ("how you doin", "how_are_you"),
            ("what's up", "how_are_you"), ("how is it going", "how_are_you"),
            ("how are things", "how_are_you"), ("how have you been", "how_are_you"),
            
            # Name queries
            ("what's your name", "name_query"), ("who are you", "name_query"),
            ("what are you", "name_query"), ("your name", "name_query"),
            ("what do i call you", "name_query"), ("tell me about yourself", "name_query"),
            
            # Help requests
            ("help", "help_request"), ("can you help", "help_request"),
            ("i need help", "help_request"), ("how do i", "help_request"),
            ("what should i do", "help_request"), ("can you assist", "help_request"),
            ("i have a problem", "help_request"), ("i need advice", "help_request"),
            
            # Sad emotions
            ("i'm sad", "sad_emotion"), ("i feel depressed", "sad_emotion"),
            ("i'm unhappy", "sad_emotion"), ("i'm upset", "sad_emotion"),
            ("feeling down", "sad_emotion"), ("not good", "sad_emotion"),
            ("i'm crying", "sad_emotion"), ("i'm heartbroken", "sad_emotion"),
            ("i'm lonely", "sad_emotion"), ("i'm stressed", "sad_emotion"),
            
            # Happy emotions
            ("i'm happy", "happy_emotion"), ("i'm excited", "happy_emotion"),
            ("great news", "happy_emotion"), ("awesome", "happy_emotion"),
            ("amazing", "happy_emotion"), ("i'm thrilled", "happy_emotion"),
            ("i'm so happy", "happy_emotion"), ("celebrating", "happy_emotion"),
            
            # Joke requests
            ("tell me a joke", "joke_request"), ("joke", "joke_request"),
            ("make me laugh", "joke_request"), ("something funny", "joke_request"),
            ("i need a laugh", "joke_request"), ("humor me", "joke_request"),
            
            # Gratitude
            ("thank you", "gratitude"), ("thanks", "gratitude"),
            ("thank you so much", "gratitude"), ("i appreciate it", "gratitude"),
            ("you're the best", "gratitude"), ("many thanks", "gratitude"),
            
            # Time
            ("what time is it", "time_query"), ("what's the time", "time_query"),
            ("time", "time_query"), ("current time", "time_query"),
            ("what hour is it", "time_query"), ("clock", "time_query"),
            
            # Weather
            ("what's the weather", "weather_query"), ("how's the weather", "weather_query"),
            ("is it hot", "weather_query"), ("is it cold", "weather_query"),
            ("will it rain", "weather_query"), ("sunny today", "weather_query"),
            
            # Affection
            ("i love you", "affection"), ("love you", "affection"),
            ("i like you", "affection"), ("you're great", "affection"),
            ("you're amazing", "affection"), ("you're awesome", "affection"),
            
            # General chat (catch-all training examples)
            ("i went to the store today", "general_chat"),
            ("i'm thinking about", "general_chat"),
            ("what do you think about", "general_chat"),
            ("i believe that", "general_chat"),
            ("in my opinion", "general_chat"),
            ("i was wondering", "general_chat"),
            ("can we talk about", "general_chat"),
            ("i have a question", "general_chat"),
        ]
        
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        if os.path.exists(self.model_path):
            print("🧠 Loading existing neural network model...")
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            print(f"✅ Model loaded! Intents: {len(self.intents)}")
        else:
            print("🧠 Training neural network intent classifier...")
            self._train_model()
    
    def _train_model(self):
        """Train the neural network"""
        # Prepare training data
        texts = [item[0] for item in self.training_data]
        labels = [item[1] for item in self.training_data]
        
        # Create pipeline: TF-IDF vectorizer + MLP Neural Network
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # Use unigrams and bigrams
                max_features=500
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32),  # Smaller layers for small dataset
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                alpha=0.01,  # Regularization
                learning_rate_init=0.001,
                early_stopping=False,  # Disable for small dataset
                warm_start=False
            ))
        ])
        
        # Train the neural network
        self.pipeline.fit(texts, labels)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"✅ Neural network trained! Architecture: {self.pipeline.named_steps['mlp'].hidden_layer_sizes}")
        print(f"   Training samples: {len(texts)}, Intents: {len(set(labels))}")
    
    def predict_intent(self, text: str) -> str:
        """Predict the intent of user message using neural network"""
        if not text or len(text.strip()) < 2:
            return 'general_chat'
        
        # Predict with neural network
        intent = self.pipeline.predict([text.lower()])[0]
        
        # Get confidence scores
        proba = self.pipeline.predict_proba([text.lower()])[0]
        confidence = np.max(proba)
        
        print(f"🧠 NN Prediction: intent='{intent}', confidence={confidence:.2f}")
        
        # If confidence is too low, fall back to general chat
        if confidence < 0.3:
            print(f"   Low confidence, using general_chat")
            return 'general_chat'
        
        return intent
    
    def get_response(self, intent: str) -> str:
        """Get a response for the predicted intent"""
        import random
        
        # Handle time specially
        if intent == 'time_query':
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            return f"It's {current_time}. Time flies when you're having fun, right?"
        
        # Get random response from intent's response pool
        if intent in self.responses:
            return random.choice(self.responses[intent])
        
        # Fallback to general chat
        return random.choice(self.responses['general_chat'])

# Global classifier instance
intent_classifier = IntentClassifier()
