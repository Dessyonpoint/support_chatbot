# 🤖 support_chatbot

An AI-powered customer support chatbot designed for intelligent query handling and automated assistance.  
SupportBot uses NLP techniques to understand user intent, generate responses, and serve them via a containerized API setup with Docker and Nginx.

---

## 🚀 Features
- 💬 **Conversational Intelligence:** Understands and responds to user queries in real time.  
- 🧠 **NLP-Powered Engine:** Uses transformer-based models for contextual responses.  
- ⚙️ **Dockerized Deployment:** Fully containerized using Docker and `docker-compose`.  
- 🌐 **Reverse Proxy Integration:** Managed via Nginx for scalable production deployment.  
- 💾 **Persistent Storage:** SQLite database (`chatbot.db`) for storing chat history or user context.

---

## 🧩 Project Structure
supportbot/
├── chatbot.py # Main chatbot API logic
├── chatbot.db # Local database
├── requirements.txt # Python dependencies
├── Dockerfile # Application container
├── docker-compose.yml # Multi-container setup
├── nginx.conf # Nginx reverse proxy config
├── .env # Environment variables
├── docker/ # Docker-related configs
└── README.md # Project documentation

---

## 🧰 Tech Stack
- **Backend:** Python (FastAPI / Flask)  
- **AI Engine:** Transformers / OpenAI API  
- **Database:** SQLite  
- **Containerization:** Docker, Docker Compose  
- **Web Server:** Nginx  

---

## ⚙️ Setup & Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Dessyonpoint/support_chatbot.git
cd support_chatbot

2️⃣ Configure Environment

Create a .env file and define necessary variables:
API_KEY=your_api_key_here
DEBUG=True

3️⃣ Build and Run with Docker

docker-compose up --build

This spins up:

The Chatbot API container

The Nginx reverse proxy

Access the app via:
https://support-chatbot-fwri.onrender.com/

🧠 API Endpoint

POST /chat
Send a user message and receive an AI-generated response.

Example:

curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "Hello!"}'

📞 Author

Desmond Andiir
LinkedIn: https://www.linkedin.com/m/in/desmond-andiir-6142a72b9/
GitHub: https://github.com/Dessyonpoint/support_chatbot

💡 SupportBot demonstrates applied NLP, API development, and scalable AI deployment using Docker — built for real-world customer support automation.


