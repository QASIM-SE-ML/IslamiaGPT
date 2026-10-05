// ---- Element references ----
const launcher = document.getElementById('launcher');
const panel = document.getElementById('panel');
const closeBtn = document.getElementById('close-btn');
const messages = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');

// ---- Open / close panel ----
launcher.addEventListener('click', () => panel.classList.toggle('open'));
closeBtn.addEventListener('click', () => panel.classList.remove('open'));

// ---- Add a message bubble to the chat window ----
function addMessage(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;

  div.textContent = text;

  messages.appendChild(div);

  messages.scrollTop = messages.scrollHeight;

  return div;
}

// ---- Send a message ----
async function sendMessage() {
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    const typingMessage = document.createElement("div");
    typingMessage.classList.add("msg", "bot");
    typingMessage.innerHTML = `
    <span class="typing-text">IslamiaGPT is typing</span>
    <span class="dots">...</span>
`;
    typingMessage.id = "typing";

    messages.appendChild(typingMessage);
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        const typing = document.getElementById("typing");
        if (typing) {
            typing.remove();
        }

        addMessage(data.reply, "bot");

    } catch (error) {

        const typing = document.getElementById("typing");
        if (typing) {
            typing.remove();
        }

        addMessage(
    "Sorry, I couldn't process your request. Please try again.",
    "bot"
);
    }
}
// ---- Event bindings ----
sendBtn.addEventListener('click', sendMessage);


input.addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

function askSuggestion(question) {
    input.value = question;
    sendMessage();
}

function clearChat() {
    messages.innerHTML = "";

    addMessage(
        "Assalam-o-Alaikum! I'm the Islamia College Peshawar assistant. How can I help you today?",
        "bot"
    );
}