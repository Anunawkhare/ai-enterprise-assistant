import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# ========== NEW: Import Gemini Library ==========
import google.generativeai as genai


# Load environment variables
load_dotenv()

# ========== NEW: Configure Gemini ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.0-pro")

# Initialize FastAPI
app = FastAPI(title="AI Enterprise Assistant", version="1.0")

# ===================== MODELS =====================
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class AskResponse(BaseModel):
    answer: str
    action_taken: Optional[str] = None
    action_result: Optional[dict] = None

# ===================== MOCK DATA =====================
MOCK_EMPLOYEES = {
    "john": {"name": "John Doe", "department": "Engineering", "email": "john.doe@company.com"},
    "jane": {"name": "Jane Smith", "department": "Marketing", "email": "jane.smith@company.com"},
    "alice": {"name": "Alice Johnson", "department": "HR", "email": "alice.johnson@company.com"},
}

MOCK_TICKETS = []
TICKET_ID = 1

# ===================== BUSINESS ACTIONS =====================
def create_ticket(description: str) -> dict:
    """Mock function to create a support ticket."""
    global TICKET_ID
    ticket = {
        "ticket_id": TICKET_ID,
        "description": description,
        "status": "open",
        "priority": "medium",
        "created_at": "2026-07-03 10:00:00"
    }
    MOCK_TICKETS.append(ticket)
    TICKET_ID += 1
    return ticket

def fetch_employee(name: str) -> dict:
    """Mock function to fetch employee details."""
    name_lower = name.lower()
    if name_lower in MOCK_EMPLOYEES:
        return MOCK_EMPLOYEES[name_lower]
    return {"error": f"Employee '{name}' not found"}

# ===================== CONVERSATION MEMORY =====================
conversation_memory = {}

def get_memory(session_id: str) -> List[dict]:
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    return conversation_memory[session_id]

def update_memory(session_id: str, user_msg: str, assistant_msg: str):
    memory = get_memory(session_id)
    memory.append({"role": "user", "content": user_msg})
    memory.append({"role": "assistant", "content": assistant_msg})
    # Keep only last 5 exchanges (10 messages)
    if len(memory) > 10:
        conversation_memory[session_id] = memory[-10:]

# ===================== LLM PROCESSING (Updated for Gemini) =====================
def process_question(question: str, session_id: str) -> dict:
    """
    Process user question using Gemini LLM and decide business action.
    """
    memory = get_memory(session_id)
    
    # Build system prompt
    system_prompt = """
    You are an AI Enterprise Assistant. Your job is to:
    1. Answer general questions professionally.
    2. Perform business actions when asked:
       - Create a ticket (detect phrases like "create ticket", "raise ticket", "log issue", "open ticket")
       - Fetch employee info (detect phrases like "find employee", "get employee", "who is", "employee details")
    
    If the user asks to create a ticket, extract the description.
    If the user asks to fetch an employee, extract the name.
    
    Respond in JSON format:
    {
        "action": "answer" | "create_ticket" | "fetch_employee",
        "response": "Your response to the user",
        "parameters": {"description": "..."} or {"name": "..."}
    }
    """
    
    # Prepare messages with memory (Gemini uses a different format)
    # We'll combine system prompt, memory, and new question into a single prompt
    prompt_text = system_prompt + "\n\n"
    
    # Add memory context
    for msg in memory:
        role = msg["role"]
        content = msg["content"]
        prompt_text += f"{role}: {content}\n"
    
    # Add the current question
    prompt_text += f"user: {question}\n"
    prompt_text += "assistant: "
    
    # Call Gemini
    try:
        response = model.generate_content(prompt_text)
        
        # Gemini returns plain text, we need to parse JSON from it
        # Try to extract JSON from the response
        response_text = response.text.strip()
        
        # Find JSON in the response (handle possible extra text)
        # Look for content between { and }
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            result = json.loads(json_str)
            return result
        else:
            # If no JSON, treat as a regular answer
            return {
                "action": "answer",
                "response": response_text,
                "parameters": {}
            }
            
    except Exception as e:
        # Fallback for error handling
        return {
            "action": "answer",
            "response": f"I'm sorry, I encountered an error processing your request. Please try again. (Error: {str(e)})",
            "parameters": {}
        }

# ===================== API ENDPOINT =====================
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # Step 1: Process the question
        result = process_question(request.question, request.session_id)
        
        action = result.get("action", "answer")
        response_text = result.get("response", "I couldn't process your request.")
        parameters = result.get("parameters", {})
        
        action_taken = None
        action_result = None
        
        # Step 2: Execute business action
        if action == "create_ticket":
            description = parameters.get("description", "No description provided")
            ticket = create_ticket(description)
            action_taken = "create_ticket"
            action_result = ticket
            response_text = f"✅ Ticket #{ticket['ticket_id']} created successfully!"
            
        elif action == "fetch_employee":
            name = parameters.get("name", "")
            employee = fetch_employee(name)
            action_taken = "fetch_employee"
            action_result = employee
            if "error" in employee:
                response_text = employee["error"]
            else:
                response_text = f"👤 **{employee['name']}** | Department: {employee['department']} | Email: {employee['email']}"
        
        # Step 3: Update memory
        update_memory(request.session_id, request.question, response_text)
        
        # Step 4: Return response
        return AskResponse(
            answer=response_text,
            action_taken=action_taken,
            action_result=action_result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ===================== HEALTH CHECK =====================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Enterprise Assistant is running!"}

# ===================== RUN =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)