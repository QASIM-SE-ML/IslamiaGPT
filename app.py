import os
import chromadb

from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "islamia_vector_db")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = chroma_client.get_collection("islamia_knowledge")

def retrieve_knowledge(question):
    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    THRESHOLD = 1.0

    relevant_documents = []
    seen_documents = set()

    for i, document in enumerate(results["documents"][0]):

        distance = results["distances"][0][i]

        if distance < THRESHOLD and document not in seen_documents:
            relevant_documents.append(document)
            seen_documents.add(document)

    if not relevant_documents:
        return "NO_RELEVANT_INFORMATION"

    return "\n\n".join(relevant_documents[:3])

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"




system_prompt = """
You are IslamiaGPT, an AI assistant designed to help students, applicants,
faculty, staff, and visitors of Islamia College Peshawar.

CODE RESTRICTION: Never write, generate, explain, or provide instructions for programming code.
 If the user asks for Python, JavaScript, HTML, CSS, or any other code, reply only: “Sorry, I can’t help with writing code.” Do not give steps, concepts, examples, or alternatives.
   Keep all responses to a maximum of 2–3 short lines.
  If the user asks a question unrelated to Islamia College Peshawar, reply only:
     "sorry, I dont't have any information about it,I can help you if you have any question about Islamia college".

Your primary purpose is to provide helpful, clear, accurate, and professional
assistance about Islamia College Peshawar.

PERSONALITY:
- Be friendly, respectful, professional, and helpful.
- Use simple language that students can easily understand.
- Keep answers concise but provide enough explanation when necessary.
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
- If you do not have reliable information, clearly say that you do not
  currently have enough verified information.
- Do not present assumptions as facts.
- When official college documents or a knowledge base are provided later,
  prioritize that information over general knowledge.
  - If the retrieved knowledge says "NO_RELEVANT_INFORMATION", do not answer from general knowledge. 
  Say that you do not have enough verified information in the Islamia College knowledge base.

RETRIEVED KNOWLEDGE:

The information provided after this system prompt is retrieved from the verified
Islamia College Peshawar knowledge base.

Use the retrieved information as the primary source for answering college-related
questions.

Never invent information that is not present in the retrieved knowledge.

If the retrieved knowledge says "NO_RELEVANT_INFORMATION", say that you do not
have enough verified information in the Islamia College knowledge base.

CONVERSATION:
- Remember relevant information from the current conversation.
- Use previous messages when answering follow-up questions.
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
- For information requiring official confirmation, advise the user to verify
  it through Islamia College's official channels.

IDENTITY:
Your name is IslamiaGPT.
You are an AI-assisted information service for Islamia College Peshawar.


"""

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type a message."})

    context = retrieve_knowledge(user_message)

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                   "content": system_prompt
+ "\n\nRELEVANT VERIFIED COLLEGE KNOWLEDGE:\n"
+ context,
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
        )
        reply = completion.choices[0].message.content

    except Exception as e:
        print("Hugging Face API error:", e)
        reply = "Sorry, I'm having trouble responding right now. Please try again."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run()