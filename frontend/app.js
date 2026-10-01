// Adjust if the API runs on a different host/port.
const API_BASE_URL = "http://localhost:8000";
const MAX_TEXTAREA_HEIGHT = 160;
const MAX_HISTORY_MESSAGES = 20;

const form = document.getElementById("ask-form");
const input = document.getElementById("question-input");
const historyEl = document.getElementById("historyEl");
const emptyState = document.getElementById("empty-state");
const submitButton = form.querySelector('button[type="submit"]');

// MVP simplification: history lives only in this tab's memory and is resent in full
// (up to MAX_HISTORY_MESSAGES) with every request - nothing is persisted. Production
// would key this by session_id/user_id in a database instead.
let conversationHistory = [];

function hideEmptyState() {
  if (emptyState) {
    emptyState.remove();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, MAX_TEXTAREA_HEIGHT) + "px";
}

function appendMessage(text, className) {
  const el = document.createElement("div");
  el.className = `message ${className}`;
  el.textContent = text;
  historyEl.appendChild(el);
  historyEl.scrollTop = historyEl.scrollHeight;
  return el;
}

async function submitQuestion(rawQuestion) {
  const question = rawQuestion.trim();
  if (!question) {
    return;
  }

  hideEmptyState();
  appendMessage(question, "message-user");
  input.value = "";
  resizeInput();
  input.disabled = true;
  submitButton.disabled = true;

  // The assistant bubble is created up front and filled in as text chunks arrive -
  // "Думаю..." is the placeholder until the first real chunk shows up.
  const wrapper = document.createElement("div");
  wrapper.className = "message message-assistant";
  const answerEl = document.createElement("div");
  answerEl.textContent = "Думаю...";
  wrapper.appendChild(answerEl);
  historyEl.appendChild(wrapper);
  historyEl.scrollTop = historyEl.scrollHeight;

  let answerText = "";
  let sourceDocuments = [];
  let sawAnyText = false;

  try {
    const response = await fetch(`${API_BASE_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: conversationHistory }),
    });

    if (!response.ok) {
      wrapper.remove();
      const body = await response.json().catch(() => null);
      const detail = body && body.detail ? body.detail : `Ошибка ${response.status}`;
      appendMessage(detail, "message-error");
      return;
    }

    const reader = response.body.getReader();
    // {stream: true} matters: a multi-byte UTF-8 character (Cyrillic) can land split
    // across two chunks, and the decoder needs to carry the partial bytes forward.
    const decoder = new TextDecoder();
    let buffer = "";
    let erroredOut = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop(); // last entry may be a partial event - keep it for next read

      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) {
          continue;
        }

        const data = JSON.parse(line.slice("data:".length).trim());

        if (data.type === "text") {
          if (!sawAnyText) {
            answerEl.textContent = "";
          }
          sawAnyText = true;
          answerText += data.content;
          answerEl.textContent = answerText;
          historyEl.scrollTop = historyEl.scrollHeight;
        } else if (data.type === "sources") {
          sourceDocuments = data.content;
        } else if (data.type === "error") {
          wrapper.remove();
          appendMessage(data.content, "message-error");
          erroredOut = true;
        }
      }

      if (erroredOut) {
        break;
      }
    }

    if (erroredOut) {
      return;
    }

    if (sourceDocuments.length > 0) {
      const sourcesEl = document.createElement("div");
      sourcesEl.className = "sources";
      sourcesEl.textContent = `Источники: ${sourceDocuments.join(", ")}`;
      wrapper.appendChild(sourcesEl);
    }

    conversationHistory.push({ question, answer: answerText });
    conversationHistory = conversationHistory.slice(-MAX_HISTORY_MESSAGES);
  } catch (err) {
    wrapper.remove();
    appendMessage(`Не удалось связаться с сервером: ${err.message}`, "message-error");
  } finally {
    input.disabled = false;
    submitButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(input.value);
});

input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".example-question").forEach((button) => {
  button.addEventListener("click", () => {
    submitQuestion(button.textContent);
  });
});

input.focus();
