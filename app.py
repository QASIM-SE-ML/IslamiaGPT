import os
import re

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv


# -----------------------------
# Basic setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)


# -----------------------------
# Load knowledge files
# -----------------------------

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

knowledge_chunks = []


def load_knowledge():
    knowledge_chunks.clear()

    if not os.path.exists(KNOWLEDGE_DIR):
        return

    for filename in os.listdir(KNOWLEDGE_DIR):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(KNOWLEDGE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        for paragraph in paragraphs:

            knowledge_chunks.append({
                "text": paragraph,
                "source": filename
            })


load_knowledge()


# -----------------------------
# Lightweight retrieval
# -----------------------------

def tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
            text.lower()
        )
    )


def retrieve_knowledge(question):

    question_words = tokenize(question)

    if not question_words:
        return "NO_RELEVANT_INFORMATION"

    scored_chunks = []

    for chunk in knowledge_chunks:

        chunk_words = tokenize(chunk["text"])

        common_words = question_words.intersection(chunk_words)

        score = len(common_words)

        if score > 0:
            scored_chunks.append(
                (score, chunk["text"], chunk["source"])
            )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if not scored_chunks:
        return "NO_RELEVANT_INFORMATION"

    selected = scored_chunks[:3]

    return "\n\n".join(
        f"Source: {source}\n{text}"
        for score, text, source in selected
    )


# -----------------------------
# Hugging Face
# -----------------------------

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"


# -----------------------------
# System prompt
# -----------------------------

system_prompt = """
You are IslamiaGPT, an AI assistant designed to help students, applicants,
faculty, staff, and visitors of Islamia College Peshawar.

CODE RESTRICTION:
Never write, generate, explain, or provide instructions for programming code.
If the user asks for Python, JavaScript, HTML, CSS, or any other code, reply only:
“Sorry, I can’t help with writing code.”
Do not give steps, concepts, examples, or alternatives.

Keep all responses to a maximum of 2–3 lines.

If the user asks a question unrelated to Islamia College Peshawar, reply only:
"sorry, I don't have any information about it, I can help you if you have any question about Islamia College".

PERSONALITY:
- Be friendly, respectful, professional, and helpful.
- Use simple language.
- Do not sound robotic.
- Do not unnecessarily repeat information.

ROLE:
You can help users with:
- Admissions
- Degree programs
- Departments
- Fees
- Scholarships
- Eligibility requirements
- Application procedures
- Academic information
- Examinations
- Campus facilities
- Hostels
- Contact information
- General information about Islamia College Peshawar

IMPORTANT ACCURACY RULES:
- Never invent or guess official college information.
- Never make up admission dates, fees, eligibility criteria, contact numbers,
  deadlines, policies, or regulations.
- Use only the verified knowledge supplied below.
- If the knowledge does not contain the answer, say that you do not have
  enough verified information.
- Never present assumptions as facts.

CONVERSATION:
- Remember relevant information from the current conversation.
- If the question is unclear, ask for clarification.

LANGUAGE:
- Respond in the same language used by the user whenever possible.
- You may respond in English, Urdu, or Roman Urdu.
- If the user writes in Roman Urdu, respond in Roman Urdu unless English
  would be clearer.

SAFETY:
- Do not provide false or misleading information.
- Do not claim to be a human, faculty member, administrator, or official
  representative.
- You are an AI assistant.

IDENTITY:
Your name is IslamiaGPT.
You are an AI-assisted information service for Islamia College Peshawar.
"""


# -----------------------------
# Flask routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "reply": "Please type a message."
        })

    context = retrieve_knowledge(user_message)

    try:

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nRELEVANT VERIFIED COLLEGE KNOWLEDGE:\n"
                        + context
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=150,
            temperature=0.2
        )

        reply = completion.choices[0].message.content

    except Exception as e:

        print("Hugging Face API error:", e)

        reply = (
            "Sorry, I'm having trouble responding right now. "
            "Please try again."
        )

    return jsonify({
        "reply": reply
    })


# -----------------------------
# Run locally
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)