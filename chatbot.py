"""
Advanced AI Support Chatbot for Paid Applications
================================================
A comprehensive chatbot system with NLP, multi-language support, file handling,
authentication, analytics, and extensive integration capabilities.

Requirements:
pip install fastapi uvicorn python-multipart sqlalchemy pydantic
pip install openai anthropic transformers sentence-transformers
pip install google-auth google-auth-oauthlib google-auth-httplib2
pip install stripe python-jose passlib bcrypt
pip install langdetect textblob redis python-dotenv
pip install aiofiles websockets
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import json
import asyncio
import logging
from enum import Enum
import uuid
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default="user")  # user, admin, moderator
    google_id = Column(String, unique=True, nullable=True)
    preferences = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    conversations = relationship("Conversation", back_populates="user")
    sessions = relationship("ChatSession", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # user, assistant, system
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sentiment = Column(String, nullable=True)
    intent = Column(String, nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5 or thumbs up/down
    has_attachment = Column(Boolean, default=False)
    attachment_path = Column(String, nullable=True)
    
    conversation = relationship("Conversation", back_populates="messages")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_token = Column(String, unique=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="sessions")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    keywords = Column(Text)  # JSON array
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Analytics(Base):
    __tablename__ = "analytics"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)
    user_id = Column(Integer, nullable=True)
    session_id = Column(String, nullable=True)
    data = Column(Text)  # JSON
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class ToneStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str


class UserPreferences(BaseModel):
    language: str = "en"
    tone: ToneStyle = ToneStyle.FRIENDLY
    theme: str = "light"
    notifications_enabled: bool = True


class Token(BaseModel):
    access_token: str
    token_type: str


class MessageRequest(BaseModel):
    content: str
    session_id: Optional[str] = None
    conversation_id: Optional[int] = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    timestamp: datetime
    sentiment: Optional[str]
    intent: Optional[str]
    quick_replies: Optional[List[str]] = None


class ConversationResponse(BaseModel):
    id: int
    session_id: str
    title: Optional[str]
    created_at: datetime
    message_count: int


# ============================================================================
# AUTHENTICATION & SECURITY
# ============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# ============================================================================
# AI CHATBOT ENGINE
# ============================================================================

class ChatbotEngine:
    """
    Core AI engine for the chatbot with NLP capabilities
    """
    
    def __init__(self):
        self.context_memory = {}  # Session-based context
        self.intents = self._load_intents()
        
    def _load_intents(self) -> Dict[str, List[str]]:
        """Load predefined intents and patterns"""
        return {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "help": ["help", "assist", "support", "need help", "how to"],
            "technical": ["error", "bug", "crash", "not working", "issue", "problem"],
            "billing": ["payment", "invoice", "subscription", "charge", "refund"],
            "account": ["account", "profile", "settings", "password", "login"],
            "feedback": ["feedback", "suggestion", "improve", "feature request"],
            "farewell": ["bye", "goodbye", "see you", "thanks", "thank you"]
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            from langdetect import detect
            return detect(text)
        except:
            return "en"
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of the message"""
        try:
            from textblob import TextBlob
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.3:
                return "positive"
            elif polarity < -0.3:
                return "negative"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def detect_intent(self, text: str) -> str:
        """Detect user intent from message"""
        text_lower = text.lower()
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent
        
        return "general_query"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities from text"""
        entities = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "numbers": re.findall(r'\b\d+\b', text),
        }
        return entities
    
    def generate_response(self, message: str, context: Dict, preferences: UserPreferences, 
                         knowledge_base: List[KnowledgeBase]) -> Dict[str, Any]:
        """
        Generate AI response based on message and context
        This is a template - integrate with your preferred AI API (OpenAI, Anthropic, etc.)
        """
        
        intent = self.detect_intent(message)
        sentiment = self.analyze_sentiment(message)
        entities = self.extract_entities(message)
        language = self.detect_language(message)
        
        # Search knowledge base
        relevant_kb = self._search_knowledge_base(message, knowledge_base)
        
        # Generate response based on intent
        response_text = self._generate_context_aware_response(
            message, intent, sentiment, context, preferences, relevant_kb
        )
        
        # Generate quick replies
        quick_replies = self._generate_quick_replies(intent)
        
        return {
            "response": response_text,
            "intent": intent,
            "sentiment": sentiment,
            "entities": entities,
            "language": language,
            "quick_replies": quick_replies,
            "suggestions": self._generate_suggestions(intent, context)
        }
    
    def _search_knowledge_base(self, query: str, kb_entries: List[KnowledgeBase]) -> List[KnowledgeBase]:
        """Search knowledge base for relevant entries"""
        query_lower = query.lower()
        relevant = []
        
        for entry in kb_entries:
            keywords = json.loads(entry.keywords) if entry.keywords else []
            if any(keyword.lower() in query_lower for keyword in keywords):
                relevant.append(entry)
                entry.usage_count += 1
        
        return relevant[:3]  # Top 3 matches
    
    def _generate_context_aware_response(self, message: str, intent: str, sentiment: str,
                                        context: Dict, preferences: UserPreferences,
                                        kb_entries: List[KnowledgeBase]) -> str:
        """Generate response adapted to context and preferences"""
        
        # Use knowledge base if available
        if kb_entries:
            return kb_entries[0].answer
        
        # Default responses by intent
        responses = {
            "greeting": f"Hello! How can I help you today?",
            "help": "I'm here to assist you! What do you need help with?",
            "technical": "I understand you're experiencing a technical issue. Could you provide more details?",
            "billing": "I can help with billing questions. What specifically would you like to know?",
            "account": "I can assist with account-related matters. What do you need?",
            "feedback": "Thank you for your feedback! We value your input.",
            "farewell": "Thank you for chatting! Feel free to reach out anytime."
        }
        
        base_response = responses.get(intent, "I'm here to help. Could you provide more details?")
        
        # Adapt tone based on preferences
        if preferences.tone == ToneStyle.PROFESSIONAL:
            return base_response
        elif preferences.tone == ToneStyle.CASUAL:
            return base_response.replace("!", ".").lower()
        else:  # FRIENDLY
            return base_response + " 😊"
    
    def _generate_quick_replies(self, intent: str) -> List[str]:
        """Generate suggested quick replies"""
        quick_replies = {
            "greeting": ["Get Help", "View Account", "Contact Support"],
            "help": ["Technical Issue", "Billing Question", "Account Settings"],
            "technical": ["Describe Issue", "Send Screenshot", "View Documentation"],
            "billing": ["View Invoice", "Update Payment", "Cancel Subscription"],
            "account": ["Change Password", "Update Profile", "Security Settings"],
        }
        return quick_replies.get(intent, ["Main Menu", "Contact Support"])
    
    def _generate_suggestions(self, intent: str, context: Dict) -> List[str]:
        """Generate proactive suggestions"""
        suggestions = []
        
        if intent == "technical":
            suggestions.append("Check our troubleshooting guide")
            suggestions.append("View system status")
        elif intent == "billing":
            suggestions.append("View pricing plans")
            suggestions.append("Download past invoices")
        
        return suggestions


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Advanced AI Support Chatbot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot engine
chatbot_engine = ChatbotEngine()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    async def send_typing_indicator(self, session_id: str, is_typing: bool):
        await self.send_message({"type": "typing", "is_typing": is_typing}, session_id)


manager = ConnectionManager()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    preferences = UserPreferences().dict()
    
    new_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        full_name=user.full_name,
        preferences=json.dumps(preferences)
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat/message")
async def send_message(
    message: MessageRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response"""
    
    # Get or create conversation
    if message.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == message.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
    else:
        session_id = message.session_id or str(uuid.uuid4())
        conversation = Conversation(
            session_id=session_id,
            user_id=current_user.id,
            title=message.content[:50]
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=message.content,
        sentiment=chatbot_engine.analyze_sentiment(message.content),
        intent=chatbot_engine.detect_intent(message.content)
    )
    db.add(user_message)
    db.commit()
    
    # Get user preferences
    preferences = UserPreferences(**json.loads(current_user.preferences))
    
    # Get knowledge base
    kb_entries = db.query(KnowledgeBase).all()
    
    # Get conversation context
    recent_messages = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.timestamp.desc()).limit(10).all()
    
    context = {
        "recent_messages": [{"role": m.role, "content": m.content} for m in recent_messages],
        "user_id": current_user.id,
        "conversation_id": conversation.id
    }
    
    # Generate AI response
    ai_result = chatbot_engine.generate_response(
        message.content, context, preferences, kb_entries
    )
    
    # Save AI response
    ai_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=ai_result["response"],
        sentiment=ai_result["sentiment"],
        intent=ai_result["intent"]
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)
    
    # Log analytics
    background_tasks.add_task(
        log_analytics, "message_sent", current_user.id, conversation.session_id,
        {"intent": ai_result["intent"], "sentiment": ai_result["sentiment"]}, db
    )
    
    return {
        "message": MessageResponse(
            id=ai_message.id,
            role=ai_message.role,
            content=ai_message.content,
            timestamp=ai_message.timestamp,
            sentiment=ai_message.sentiment,
            intent=ai_message.intent,
            quick_replies=ai_result["quick_replies"]
        ),
        "conversation_id": conversation.id,
        "suggestions": ai_result["suggestions"]
    }


@app.post("/chat/upload")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: int = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload file attachment"""
    import aiofiles
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Save file
    file_path = f"uploads/{current_user.id}/{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Create message with attachment
    if conversation_id:
        message = Message(
            conversation_id=conversation_id,
            role="user",
            content=f"[Uploaded: {file.filename}]",
            has_attachment=True,
            attachment_path=file_path
        )
        db.add(message)
        db.commit()
    
    return {"filename": file.filename, "path": file_path, "size": len(content)}


@app.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's conversations"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.is_archived == False
    ).order_by(Conversation.updated_at.desc()).all()
    
    return [
        ConversationResponse(
            id=conv.id,
            session_id=conv.session_id,
            title=conv.title,
            created_at=conv.created_at,
            message_count=len(conv.messages)
        )
        for conv in conversations
    ]


@app.get("/conversation/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages from a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp).all()
    
    return [
        MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            sentiment=msg.sentiment,
            intent=msg.intent
        )
        for msg in messages
    ]


@app.post("/message/{message_id}/rate")
async def rate_message(
    message_id: int,
    rating: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rate a message (thumbs up/down or 1-5 stars)"""
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    message.rating = rating
    db.commit()
    
    return {"status": "success", "rating": rating}


@app.get("/analytics/dashboard")
async def get_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analytics dashboard data (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    total_conversations = db.query(Conversation).count()
    total_messages = db.query(Message).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    
    # Get intent distribution
    messages = db.query(Message).filter(Message.intent != None).all()
    intent_counts = {}
    for msg in messages:
        intent_counts[msg.intent] = intent_counts.get(msg.intent, 0) + 1
    
    # Get sentiment distribution
    sentiment_counts = {}
    for msg in messages:
        if msg.sentiment:
            sentiment_counts[msg.sentiment] = sentiment_counts.get(msg.sentiment, 0) + 1
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "active_users": active_users,
        "intent_distribution": intent_counts,
        "sentiment_distribution": sentiment_counts
    }


@app.post("/knowledge-base")
async def add_knowledge(
    category: str,
    question: str,
    answer: str,
    keywords: List[str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add to knowledge base (admin/moderator only)"""
    if current_user.role not in ["admin", "moderator"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    kb_entry = KnowledgeBase(
        category=category,
        question=question,
        answer=answer,
        keywords=json.dumps(keywords)
    )
    db.add(kb_entry)
    db.commit()
    
    return {"status": "success", "id": kb_entry.id}


@app.get("/export/conversation/{conversation_id}")
async def export_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export conversation history"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp).all()
    
    export_data = {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "created_at": str(conversation.created_at),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": str(msg.timestamp),
                "sentiment": msg.sentiment
            }
            for msg in messages
        ]
    }
    
    return export_data


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Send typing indicator
            await manager.send_typing_indicator(session_id, True)
            
            # Process message (simplified - integrate with main chat logic)
            await asyncio.sleep(1)  # Simulate processing
            
            response = {
                "type": "message",
                "role": "assistant",
                "content": "This is a real-time response",
                "timestamp": str(datetime.utcnow())
            }
            
            await manager.send_typing_indicator(session_id, False)
            await manager.send_message(response, session_id)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(session_id)


def log_analytics(event_type: str, user_id: int, session_id: str, data: Dict, db: Session):
    """Background task to log analytics"""
    analytics = Analytics(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        data=json.dumps(data)
    )
    db.add(analytics)
    db.commit()

@app.get("/")
def home():
    return {
        "message": "Support Chatbot API is running!",
        "docs": "/docs",
        "health": "/health"
    }
# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🤖 Advanced AI Support Chatbot started successfully!")
    logger.info("📊 Database initialized")
    logger.info("🔐 Authentication system ready")
    logger.info("🌐 WebSocket connections enabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
