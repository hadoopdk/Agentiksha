"# Agentiksha" 
# Agentic AI - Voice Companion

A voice-only conversational AI that talks to you like a human using speech recognition and synthesis.

## Features

- 🎤 **Voice Recognition** - Speak naturally, the AI listens and understands
- 🔊 **Voice Synthesis** - AI responds with natural-sounding speech
- 🧠 **Human-like Personality** - Context-aware, empathetic, and conversational responses
- ⚡ **Real-time** - WebSocket connection for instant back-and-forth conversation
- 🎨 **Beautiful Interface** - Modern UI with visual feedback for listening/speaking states

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python voice_main.py
```

### 3. Open in Browser

Navigate to `http://localhost:8000` in your web browser.

**Recommended Browsers:** Chrome, Edge, or Safari for best speech recognition support.

### 4. Start Talking

1. Click the **"Start Talking"** button
2. Allow microphone access when prompted
3. Speak naturally - pause briefly when done
4. The AI will listen, think, and respond with voice!

## How It Works

1. **Speech Recognition** (Browser's Web Speech API) captures your voice and converts it to text
2. **WebSocket** sends your message to the Python backend in real-time
3. **AI Brain** (FastAPI) generates contextual, human-like responses
4. **Speech Synthesis** (Browser's Speech Synthesis API) speaks the AI's response
5. **Visual Feedback** shows listening/speaking/thinking states with animations

## Project Structure

```
agenticai/
├── voice_main.py      # FastAPI backend with WebSocket and AI logic
├── static/
│   └── index.html     # Voice interface frontend
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Customization

### Change AI Voice
- Use the "AI Voice" dropdown to select different voices (OS-dependent)
- Adjust "Speech Speed" to make the AI speak faster or slower

### Conversation Topics
The AI can discuss:
- Casual greetings and small talk
- Help with questions and advice
- Emotional support (sad/happy moments)
- Jokes and humor
- Time and general information
- ...and much more!

## Requirements

- Python 3.8+
- Modern web browser with Web Speech API support (Chrome/Edge/Safari)
- Microphone access

## Technologies

- **Backend:** FastAPI, WebSockets, Python
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Speech:** Web Speech API (SpeechRecognition + SpeechSynthesis)

## Notes

- The AI maintains conversation memory per session
- All speech processing happens in your browser (privacy-friendly)
- Requires an internet connection for the best speech recognition accuracy
- The AI has a built-in personality designed to be friendly and engaging

## Troubleshooting

**Microphone not working?**
- Check browser permissions and allow microphone access
- Ensure your microphone is properly connected
- Try refreshing the page

**Speech recognition not accurate?**
- Speak clearly and at a moderate pace
- Reduce background noise
- Check your internet connection

**Browser not supported?**
- Use Chrome, Edge, or Safari for full functionality
- Firefox may have limited speech recognition support
