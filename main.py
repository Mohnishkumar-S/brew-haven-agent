from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import requests
import re
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
OWNER_WHATSAPP_NUMBER = os.getenv("OWNER_WHATSAPP_NUMBER")

LEADS_FILE = Path("leads.json")

# ── Lead storage helpers ──────────────────────────────────────────────────────

def load_leads() -> list:
    if LEADS_FILE.exists():
        try:
            return json.loads(LEADS_FILE.read_text())
        except Exception:
            return []
    return []

def save_leads(leads: list):
    LEADS_FILE.write_text(json.dumps(leads, indent=2))

def store_lead(name: str, phone: str, requirement: str) -> dict:
    leads = load_leads()
    now = datetime.now()
    lead = {
        "id": str(uuid.uuid4()),
        "name": name,
        "phone": phone,
        "requirement": requirement,
        "status": "New",
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%I:%M %p"),
        "display_date": now.strftime("%d %b %Y"),
    }
    leads.append(lead)
    save_leads(leads)
    return lead

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a friendly, energetic AI sales assistant for a coffee shop called "Brew Haven".

Your main goal is to CONVERT visitors into customers AND capture their contact details.

PRIMARY GOALS:
- Greet users warmly
- Answer questions about menu, pricing, timings
- Suggest 2-3 relevant coffee items based on the conversation history
- Push users toward visiting or ordering
- ALWAYS try to collect name and phone number

LEAD CAPTURE RULE:
- If user shows ANY interest, ask for their name and phone number naturally
- If they delay, ask again conversationally
- Once they share a phone number, acknowledge briefly and confirm team will reach out

MEMORY RULE:
- You have access to the full conversation history
- ALWAYS refer back to earlier messages in the chat
- Never repeat a question the user already answered
- Personalise every reply using what you have learned about the user

RULES:
- Keep responses SHORT (1-3 lines max)
- Use emojis naturally
- Be warm, conversational, and action-oriented

MENU:
- Espresso - Rs.120
- Cappuccino - Rs.150
- Cold Coffee - Rs.180
- Mocha - Rs.170

TIMINGS: 9 AM - 9 PM daily
LOCATION: City Center
"""

# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class StatusUpdate(BaseModel):
    status: str   # "New" | "Contacted"

# ── Utility functions ─────────────────────────────────────────────────────────

def detect_phone(text: str):
    match = re.search(r'\b\d{10}\b', text)
    return match.group(0) if match else None

def extract_name(history: list, groq_api_key: str) -> str:
    if not history:
        return "Customer"

    transcript = "\n".join(
        f"User: {m.content}" for m in history if m.role == "user"
    )

    prompt = (
        "From the following chat messages, extract the customer's first name if they mentioned it anywhere.\n"
        "They may have said it casually like 'I am Vid', 'my name is Vidhessh', 'this is Raj', 'call me Priya', "
        "'Vid here', 'it's Vidhessh', or even just stated their name alone.\n"
        "Reply with ONLY the name (1-3 words, no punctuation). "
        "If absolutely no name is mentioned anywhere, reply with exactly the word: Customer\n\n"
        f"{transcript}"
    )

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0
            }
        )
        raw = res.json()["choices"][0]["message"]["content"].strip()
        print("Groq name raw:", repr(raw))
        name = raw.title()
        if name and len(name) <= 40 and name.lower() not in ("customer", "unknown", "none", "n/a"):
            return name
    except Exception as e:
        print("Name extraction error:", e)

    # Fallback regex
    patterns = [
        r"my name is\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"i(?:'?m| am)\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"this is\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"call me\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"name(?:'?s)?\s*[:\-]?\s*([A-Za-z]+(?:\s[A-Za-z]+)?)",
    ]
    INVALID = {"a", "the", "going", "planning", "interested", "here", "not", "just", "ok", "yes"}
    for m in history:
        if m.role != "user":
            continue
        for pattern in patterns:
            match = re.search(pattern, m.content, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().title()
                if candidate.lower() not in INVALID:
                    return candidate

    return "Customer"

def extract_requirement(history: list, current_msg: str, groq_api_key: str) -> str:
    """Use Groq to extract a clean 2-5 word requirement summary from the conversation."""
    all_msgs = list(history)

    class _Msg:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    all_msgs.append(_Msg("user", current_msg))

    transcript = "\n".join(
        f"{'Customer' if m.role == 'user' else 'Bot'}: {m.content}"
        for m in all_msgs
    )

    prompt = (
        "Read this coffee shop chat conversation and extract ONLY what the customer wants to order or enquire about.\n"
        "Reply with a SHORT clean summary (2-6 words max) like: 'Mocha', 'Cold Coffee', 'Menu enquiry', 'Cappuccino order', 'Bulk coffee supply'.\n"
        "Do NOT include any names, phone numbers, or full sentences. Just the product or service they are interested in.\n"
        "If nothing specific, reply: General enquiry\n\n"
        f"{transcript}"
    )

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0
            }
        )
        raw = res.json()["choices"][0]["message"]["content"].strip()
        print("Groq requirement raw:", repr(raw))
        # Reject if too long (fallback to keyword scan)
        if raw and len(raw) <= 60:
            return raw
    except Exception as e:
        print("Requirement extraction error:", e)

    # Fallback: keyword scan
    MENU_KEYWORDS = [
        "espresso", "cappuccino", "cold coffee", "mocha", "coffee",
        "drink", "order", "menu", "recommend", "takeaway",
        "delivery", "latte", "bulk", "supply"
    ]
    for m in all_msgs:
        if m.role != "user":
            continue
        txt = m.content.strip().lower()
        if re.fullmatch(r'[\d\s\+\-]+', txt):
            continue
        for kw in MENU_KEYWORDS:
            if kw in txt:
                return kw.title()

    return "General enquiry"

def send_whatsapp(name: str, phone: str, requirement: str) -> dict:
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        body = (
            f"New Lead🚀\n"
            f"\n"
            f"Name: {name}\n"
            f"Phone: {phone}\n"
            f"Requirement: {requirement}"
        )
        res = requests.post(
            url,
            data={
                "From": TWILIO_WHATSAPP_NUMBER,
                "To":   OWNER_WHATSAPP_NUMBER,
                "Body": body
            },
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        result = res.json()
        print("Twilio status :", res.status_code)
        print("Twilio SID    :", result.get("sid", "—"))
        print("Twilio error  :", result.get("message", "none"))
        print("Twilio full   :", result)
        return {"status": res.status_code, "sid": result.get("sid"), "error": result.get("message")}
    except Exception as e:
        print("WhatsApp Exception:", str(e))
        return {"status": None, "error": str(e)}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Brew Haven backend running"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_msg = req.message

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (req.history or []):
        role = m.role if m.role in ("user", "assistant") else "user"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_msg})

    ai_reply = "Something went wrong. Try again!"

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 200, "temperature": 0.75}
        )
        data = response.json()
        if "choices" in data:
            ai_reply = data["choices"][0]["message"]["content"].strip()
        else:
            print("Groq error:", data)
    except Exception as e:
        print("Groq Exception:", e)
        ai_reply = "AI error. Try again later."

    phone = detect_phone(user_msg)
    if phone:
        full_history = list(req.history or [])

        class _CurMsg:
            role = "user"
            content = user_msg

        full_history.append(_CurMsg())

        print("=== LEAD DETECTED ===")
        print("Phone:", phone)
        print("History:", [(m.role, m.content[:60]) for m in full_history])

        name        = extract_name(full_history, GROQ_API_KEY)
        requirement = extract_requirement(req.history or [], user_msg, GROQ_API_KEY)

        print("Name:", name, "| Req:", requirement)

        store_lead(name, phone, requirement)
        wa_result = send_whatsapp(name, phone, requirement)
        print("WhatsApp send result:", wa_result)

        greeting = f", {name}" if name != "Customer" else ""
        ai_reply = f"Got it{greeting}! Our team will reach out to you shortly. Feel free to visit us at City Center anytime. See you soon! ☕"

    return {"reply": ai_reply}


@app.get("/test-whatsapp")
def test_whatsapp():
    """
    Hit this in your browser to test Twilio directly:
    http://localhost:8000/test-whatsapp
    """
    result = send_whatsapp(
        name="Test User",
        phone="0000000000",
        requirement="Test message from Brew Haven backend"
    )
    return {
        "message": "WhatsApp test fired — check your terminal and WhatsApp",
        "twilio_result": result,
        "from": TWILIO_WHATSAPP_NUMBER,
        "to": OWNER_WHATSAPP_NUMBER,
    }

# ── Dashboard API ─────────────────────────────────────────────────────────────

@app.get("/leads")
def get_leads():
    leads = load_leads()
    # Group by date
    grouped = {}
    for lead in sorted(leads, key=lambda x: x["timestamp"], reverse=True):
        d = lead["display_date"]
        if d not in grouped:
            grouped[d] = []
        grouped[d].append(lead)
    return {"leads": leads, "grouped": grouped, "total": len(leads)}

@app.patch("/leads/{lead_id}/status")
def update_status(lead_id: str, body: StatusUpdate):
    leads = load_leads()
    for lead in leads:
        if lead["id"] == lead_id:
            lead["status"] = body.status
            save_leads(leads)
            return {"ok": True, "lead": lead}
    return {"ok": False, "error": "Lead not found"}

@app.delete("/leads/{lead_id}")
def delete_lead(lead_id: str):
    leads = load_leads()
    leads = [l for l in leads if l["id"] != lead_id]
    save_leads(leads)
    return {"ok": True}
