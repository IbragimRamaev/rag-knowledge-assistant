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

// Scans a single line for **bold** spans and appends them as real nodes - never
// innerHTML, so anything that isn't a recognized **marker** (an HTML tag, a
// "[link](url)", a bare "#" or "`code`") is just literal text, inert by
// construction. An unclosed "**" (no closing marker yet in this line) is left as
// literal text too - renderAnswer() re-parses the full text on every chunk, so once
// the closing "**" streams in, the next call renders it correctly.
function appendBoldSpans(parentEl, line) {
  let remaining = line;
  while (remaining.length > 0) {
    const start = remaining.indexOf("**");
    if (start === -1) {
      parentEl.appendChild(document.createTextNode(remaining));
      return;
    }

    const end = remaining.indexOf("**", start + 2);
    if (end === -1) {
      parentEl.appendChild(document.createTextNode(remaining));
      return;
    }

    if (start > 0) {
      parentEl.appendChild(document.createTextNode(remaining.slice(0, start)));
    }

    const boldText = remaining.slice(start + 2, end);
    if (boldText.length > 0) {
      const strong = document.createElement("strong");
      strong.textContent = boldText;
      parentEl.appendChild(strong);
    }

    remaining = remaining.slice(end + 2);
  }
}

// Appends `text` (one paragraph/list-item's worth, possibly with internal single
// newlines) to parentEl: **bold** parsed per line, lines joined with real <br>
// elements.
function appendInlineContent(parentEl, text) {
  const lines = text.split("\n");
  lines.forEach((line, index) => {
    appendBoldSpans(parentEl, line);
    if (index < lines.length - 1) {
      parentEl.appendChild(document.createElement("br"));
    }
  });
}

// Renders the full accumulated answer text into container, from scratch, every
// time it's called (cheap at chat-message sizes - simpler and more robust against
// chunk boundaries than trying to patch the DOM incrementally). Recognizes
// blank-line-separated paragraphs, "- " bullet lists, and **bold** - nothing else.
// A list doesn't need a blank line to separate it from surrounding prose: within
// each block, runs of consecutive "- " lines become their own <ul>, runs of other
// lines become their own paragraph, and the two kinds can alternate (e.g. a
// "**Header**\n- item1\n- item2" block with no blank line becomes one paragraph
// followed by one list). Links, images, HTML, headers and tables are deliberately
// not special-cased, so they render as literal text via the same
// textContent/createTextNode path as everything else.
function renderAnswer(container, text) {
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }

  const blocks = text.split(/\n{2,}/);

  for (const block of blocks) {
    if (block.trim().length === 0) {
      continue;
    }

    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    let i = 0;
    while (i < lines.length) {
      if (lines[i].startsWith("- ")) {
        const ul = document.createElement("ul");
        ul.className = "answer-list";
        while (i < lines.length && lines[i].startsWith("- ")) {
          const li = document.createElement("li");
          appendInlineContent(li, lines[i].slice(2));
          ul.appendChild(li);
          i++;
        }
        container.appendChild(ul);
      } else {
        const paragraphLines = [];
        while (i < lines.length && !lines[i].startsWith("- ")) {
          paragraphLines.push(lines[i]);
          i++;
        }
        const p = document.createElement("div");
        p.className = "answer-paragraph";
        appendInlineContent(p, paragraphLines.join("\n"));
        container.appendChild(p);
      }
    }
  }
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
          answerText += data.content;
          renderAnswer(answerEl, answerText);
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
