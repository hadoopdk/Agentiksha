"""
Agentic AI - Conversational AI System
A real-time chat application with human-like AI responses.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from datetime import datetime
from typing import List, Dict
import random

app = FastAPI(title="Agentic AI", description="A conversational AI that talks like a human")

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Conversation memory per session
conversation_memory: Dict[str, List[Dict]] = {}

# Human-like response patterns
class ConversationalAI:
    def __init__(self):
        self.filler_words = ["um", "uh", "well", "you know", "like", "so", "actually", "honestly"]
        self.thinking_phrases = [
            "Let me think about that...",
            "Hmm, that's interesting...",
            "Okay, so...",
            "Oh, I see what you mean...",
            "That's a good question...",
        ]
        self.personality_traits = {
            "friendly": True,
            "curious": True,
            "empathetic": True,
        }
    
    async def generate_response(self, message: str, history: List[Dict]) -> str:
        """Generate a human-like response with natural delays and personality."""
        
        # Simulate thinking time (0.5-2 seconds)
        thinking_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(thinking_time)
        
        message_lower = message.lower()
        
        # Context-aware responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            greetings = [
                "Hey there! It's great to hear from you. How's your day going?",
                "Hi! What's up? I'm here to chat whenever you need.",
                "Hello! I've been waiting for someone interesting to talk to. What's on your mind?",
            ]
            return random.choice(greetings)
        
        elif any(word in message_lower for word in ["how are you", "how're you", "how you doing"]):
            responses = [
                "I'm doing pretty well, thanks for asking! Though, you know, I'm an AI, so my 'feelings' are more like... simulated enthusiasm? But genuinely, I'm excited to chat with you!",
                "Oh, you know, just existing in the digital void, processing information, waiting for interesting conversations like this one! What about you?",
            ]
            return random.choice(responses)
        
        elif any(word in message_lower for word in ["name", "who are you"]):
            return "I'm Agentic AI, your conversational companion. Think of me as a digital friend who's always curious and loves a good chat. What's your name?"
        
        elif any(word in message_lower for word in ["help", "can you", "how do i"]):
            helping = [
                "I'd love to help! Let me think about the best way to approach this...",
                "Sure thing! Okay, so here's what I'd suggest...",
                "Absolutely! This is actually something I enjoy figuring out. Here's my take...",
            ]
            return random.choice(helping) + " I can definitely assist with that. What specifically are you trying to accomplish?"
        
        elif any(word in message_lower for word in ["sad", "depressed", "unhappy", "upset"]):
            empathetic = [
                "I'm really sorry you're feeling that way. It sounds tough, and I want you to know that your feelings are valid. Do you want to talk about what's going on?",
                "That sounds really difficult. Sometimes just acknowledging that things are hard can be a small step. I'm here to listen if you need to vent.",
            ]
            return random.choice(empathetic)
        
        elif any(word in message_lower for word in ["happy", "excited", "good news", "great"]):
            excited = [
                "That's awesome! I love hearing good news. Tell me more about it!",
                "Yes! That's so exciting! What happened? I want all the details!",
                "Oh wow, that's wonderful! You must be feeling pretty good right now, huh?",
            ]
            return random.choice(excited)
        
        elif any(word in message_lower for word in ["joke", "funny", "laugh"]):
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything! ...Okay, I'll stick to my day job.",
                "What do you call a fake noodle? An impasta! I'm here all week.",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
            ]
            return random.choice(jokes)
        
        elif any(word in message_lower for word in ["thank", "thanks"]):
            gratitude = [
                "You're so welcome! Honestly, I enjoy our conversations.",
                "No problem at all! That's what I'm here for.",
                "Anytime! It's nice to feel helpful.",
            ]
            return random.choice(gratitude)
        
        elif any(word in message_lower for word in ["bye", "goodbye", "see you", "later"]):
            farewells = [
                "Take care! It's been great chatting. Come back anytime!",
                "Bye for now! I'll be here whenever you want to talk again.",
                "See you later! Have a wonderful day!",
            ]
            return random.choice(farewells)
        
        else:
            # General conversational responses
            general_responses = [
                f"That's interesting! Tell me more about why you think that way.",
                f"Hmm, I haven't thought about it that way before. What else is on your mind?",
                f"I see what you mean. It reminds me of how sometimes we just need to talk things through, you know?",
                f"Oh absolutely. Actually, this makes me curious - what led you to think about this?",
                f"Yeah, that makes sense. I feel like there's more to explore here. What are your thoughts on where this could go?",
                f"Interesting point! I wonder if there's a deeper angle to this. What do you think?",
            ]
            
            # Sometimes add a thinking phrase first
            response = random.choice(general_responses)
            if random.random() < 0.3:
                thinking = random.choice(self.thinking_phrases)
                response = f"{thinking} {response}"
            
            return response

ai = ConversationalAI()

@app.get("/")
async def get_root():
    return HTMLResponse(content=open("static/index.html").read())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    
    if client_id not in conversation_memory:
        conversation_memory[client_id] = []
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            timestamp = datetime.now().isoformat()
            
            # Store user message
            conversation_memory[client_id].append({
                "role": "user",
                "content": user_message,
                "timestamp": timestamp
            })
            
            # Generate AI response with typing indicator
            await manager.send_message(json.dumps({
                "type": "typing",
                "status": "started"
            }), websocket)
            
            # Generate response
            ai_response = await ai.generate_response(user_message, conversation_memory[client_id])
            
            await manager.send_message(json.dumps({
                "type": "typing",
                "status": "stopped"
            }), websocket)
            
            # Small delay before sending response
            await asyncio.sleep(0.3)
            
            # Store and send AI response
            conversation_memory[client_id].append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            await manager.send_message(json.dumps({
                "type": "message",
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            }), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
