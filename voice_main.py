"""
Agentic AI - Voice Conversational AI System with Neural Network
A voice-only chat application with neural network-based intent classification.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import json
import asyncio
from datetime import datetime
from typing import List, Dict
import random
import os

# Import neural network intent classifier
from intent_classifier import intent_classifier, IntentClassifier

app = FastAPI(title="Agentic AI Voice", description="A voice-only conversational AI that talks like a human")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Neural Network AI with intent classification
class NeuralVoiceAI:
    """AI using Neural Network for intent classification"""
    
    def __init__(self):
        self.classifier = intent_classifier
        self.personality_traits = {
            "friendly": True,
            "curious": True,
            "empathetic": True,
            "conversational": True,
        }
    
    async def generate_response(self, message: str, history: List[Dict]) -> str:
        """Generate response using neural network intent classification."""
        
        # Simulate neural network processing time (feels more realistic)
        thinking_time = random.uniform(0.8, 2.0)
        await asyncio.sleep(thinking_time)
        
        # Empty message check
        if not message or len(message.strip()) < 2:
            return "I didn't quite catch that. Could you say that again?"
        
        # Use neural network to predict intent
        print(f"🧠 Processing: '{message}'")
        intent = self.classifier.predict_intent(message)
        
        # Get response based on predicted intent
        response = self.classifier.get_response(intent)
        
        # Add personality variation based on conversation history
        if len(history) > 4:
            # For longer conversations, be more personal
            follow_ups = [
                " By the way, I'm enjoying our conversation!",
                " It's nice getting to know you better.",
                "",
            ]
            if random.random() < 0.3:
                response += random.choice(follow_ups)
        
        return response

ai = NeuralVoiceAI()

@app.get("/")
async def get_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    
    if client_id not in conversation_memory:
        conversation_memory[client_id] = []
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            msg_type = message_data.get("type", "message")
            
            if msg_type == "user_speech":
                user_message = message_data.get("transcript", "")
                confidence = message_data.get("confidence", 0)
                
                # LOG USER SPEECH
                print(f"\n🎤 USER SAID: {user_message}")
                print(f"   Confidence: {confidence:.2f}")
                
                if confidence < 0.3 and len(user_message) < 3:
                    print(f"   ⚠️  Low confidence, asking user to repeat")
                    await manager.send_message(json.dumps({
                        "type": "ai_response",
                        "content": "I didn't quite catch that. Could you speak a bit louder or slower?",
                        "speak": True
                    }), websocket)
                    continue
                
                timestamp = datetime.now().isoformat()
                
                # Store user message
                conversation_memory[client_id].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": timestamp
                })
                
                # Notify that AI is thinking/processing
                await manager.send_message(json.dumps({
                    "type": "ai_thinking",
                    "status": "started"
                }), websocket)
                
                # Generate response
                ai_response = await ai.generate_response(user_message, conversation_memory[client_id])
                
                await manager.send_message(json.dumps({
                    "type": "ai_thinking",
                    "status": "stopped"
                }), websocket)
                
                # Store AI response
                conversation_memory[client_id].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send response with speak flag
                await manager.send_message(json.dumps({
                    "type": "ai_response",
                    "content": ai_response,
                    "speak": True
                }), websocket)
                
                # LOG AI RESPONSE
                print(f"🤖 AI SAID: {ai_response}\n")
            
            elif msg_type == "ping":
                await manager.send_message(json.dumps({"type": "pong"}), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client {client_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
