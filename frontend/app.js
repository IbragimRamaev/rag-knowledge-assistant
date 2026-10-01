// Adjust if the API runs on a different host/port.
const API_BASE_URL = "http://localhost:8000";
const MAX_TEXTAREA_HEIGHT = 160;

const form = document.getElementById("ask-form");
const input = document.getElementById("question-input");
const historyEl = document.getElementById("historyEl");
const emptyState = document.getElementById("empty-state");
const submitButton = form.querySelector('button[type="submit"]');

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

function appendAssistantMessage(answer, sourceDocuments) {
  const wrapper = document.createElement("div");
  wrapper.className = "message message-assistant";

  const answerEl = document.createElement("div");
  answerEl.textContent = answer;
  wrapper.appendChild(answerEl);

  if (sourceDocuments && sourceDocuments.length > 0) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";
    sourcesEl.textContent = `Источники: ${sourceDocuments.join(", ")}`;
    wrapper.appendChild(sourcesEl);
  }

  historyEl.appendChild(wrapper);
  historyEl.scrollTop = historyEl.scrollHeight;
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

  const loadingEl = appendMessage("Думаю...", "message-loading");

  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    loadingEl.remove();

    const body = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = body && body.detail ? body.detail : `Ошибка ${response.status}`;
      appendMessage(detail, "message-error");
      return;
    }

    appendAssistantMessage(body.answer, body.source_documents);
  } catch (err) {
    loadingEl.remove();
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
