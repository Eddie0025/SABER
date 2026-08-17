/**
 * SABER Web UI — Main Application Controller
 */

// Application State
const state = {
  currentSessionId: null,
  sessions: [],
  isGenerating: false,
  sentinelMode: "1_sentinel", // "bolt" (no sentinel), "1_sentinel" (default), "2_sentinel" (deep thinking)
};

// DOM Elements
const elements = {
  sidebar: document.getElementById("sidebar"),
  toggleSidebarBtn: document.getElementById("toggleSidebarBtn"),
  mobileMenuBtn: document.getElementById("mobileMenuBtn"),
  newChatBtn: document.getElementById("newChatBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  historyList: document.getElementById("historyList"),
  welcomeHero: document.getElementById("welcomeHero"),
  messagesContainer: document.getElementById("messagesContainer"),
  chatStream: document.getElementById("chatStream"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  modePillBtns: document.querySelectorAll(".mode-pill-btn"),
  promptCards: document.querySelectorAll(".prompt-card"),
};

// ==========================================================================
// Initialization
// ==========================================================================

function init() {
  loadSessions();
  setupEventListeners();
  autoResizeTextarea();
}

// ==========================================================================
// Session Management
// ==========================================================================

function loadSessions() {
  try {
    const saved = localStorage.getItem("saber_chat_sessions");
    if (saved) {
      state.sessions = JSON.parse(saved);
    }
  } catch (e) {
    state.sessions = [];
  }

  if (state.sessions.length > 0) {
    renderHistory();
    // Load most recent session
    loadSession(state.sessions[0].id);
  } else {
    createNewSession();
  }
}

function saveSessions() {
  try {
    localStorage.setItem("saber_chat_sessions", JSON.stringify(state.sessions));
  } catch (e) {
    console.error("Failed to save sessions to localStorage:", e);
  }
}

function createNewSession() {
  const newSession = {
    id: "session_" + Date.now(),
    title: "New chat",
    messages: [],
    createdAt: new Date().toISOString(),
  };

  state.sessions.unshift(newSession);
  state.currentSessionId = newSession.id;
  saveSessions();
  renderHistory();
  renderCurrentChat();
}

function loadSession(sessionId) {
  state.currentSessionId = sessionId;
  renderHistory();
  renderCurrentChat();
}

function getCurrentSession() {
  return state.sessions.find(s => s.id === state.currentSessionId);
}

function renderHistory() {
  elements.historyList.innerHTML = "";

  state.sessions.forEach(session => {
    const item = document.createElement("div");
    item.className = `history-item ${session.id === state.currentSessionId ? "active" : ""}`;
    item.innerHTML = `
      <div class="history-item-title">${escapeHtml(session.title || "New chat")}</div>
    `;

    item.addEventListener("click", () => {
      if (session.id !== state.currentSessionId) {
        loadSession(session.id);
      }
    });

    elements.historyList.appendChild(item);
  });
}

function renderCurrentChat() {
  const session = getCurrentSession();
  if (!session || session.messages.length === 0) {
    elements.welcomeHero.style.display = "block";
    elements.chatStream.innerHTML = "";
    return;
  }

  elements.welcomeHero.style.display = "none";
  elements.chatStream.innerHTML = "";

  session.messages.forEach(msg => {
    appendMessageToDOM(msg.role, msg.content, msg.thinking, false);
  });

  scrollToBottom();
}

// ==========================================================================
// Message Rendering
// ==========================================================================

function appendMessageToDOM(role, content, thinking = "", animate = true) {
  elements.welcomeHero.style.display = "none";

  const row = document.createElement("div");
  row.className = `message-row ${role}-row`;

  if (role === "user") {
    row.innerHTML = `
      <div class="message-body">
        ${escapeHtml(content).replace(/\n/g, "<br>")}
      </div>
    `;
  } else {
    // Assistant message with optional thinking accordion
    let thinkingHtml = "";
    if (thinking && thinking.trim()) {
      thinkingHtml = `
        <div class="thinking-block">
          <div class="thinking-header" onclick="this.parentElement.classList.toggle('expanded')">
            <svg class="thinking-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            <span>Thinking Process</span>
          </div>
          <div class="thinking-content">${escapeHtml(thinking)}</div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="message-avatar">S</div>
      <div class="message-body">
        ${thinkingHtml}
        <div class="assistant-content">${MarkdownRenderer.render(content)}</div>
        <div class="message-actions">
          <button class="action-btn copy-msg-btn" title="Copy response">
            <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
          </button>
        </div>
      </div>
    `;

    // Setup copy action
    const copyBtn = row.querySelector(".copy-msg-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(content);
        copyBtn.style.color = "#10b981";
        setTimeout(() => {
          copyBtn.style.color = "";
        }, 1500);
      });
    }
  }

  elements.chatStream.appendChild(row);
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// ==========================================================================
// Messaging & Streaming
// ==========================================================================

async function sendMessage(text) {
  const query = text || elements.messageInput.value.trim();
  if (!query || state.isGenerating) return;

  const session = getCurrentSession();
  if (!session) return;

  // Set session title from first message
  if (session.messages.length === 0) {
    session.title = query.slice(0, 30) + (query.length > 30 ? "..." : "");
    renderHistory();
  }

  // Add user message to state & DOM
  session.messages.push({ role: "user", content: query });
  appendMessageToDOM("user", query);

  // Clear input
  elements.messageInput.value = "";
  elements.messageInput.style.height = "auto";
  elements.sendBtn.disabled = true;
  state.isGenerating = true;

  // Create temporary assistant placeholder row
  const row = document.createElement("div");
  row.className = "message-row assistant-row";
  row.innerHTML = `
    <div class="message-avatar">S</div>
    <div class="message-body">
      <div class="thinking-placeholder" id="activeThinking"></div>
      <div class="assistant-content" id="activeOutput">
        <span class="typing-cursor"></span>
      </div>
    </div>
  `;
  elements.chatStream.appendChild(row);
  scrollToBottom();

  const thinkingContainer = row.querySelector("#activeThinking");
  const outputContainer = row.querySelector("#activeOutput");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        sentinel_mode: state.sentinelMode,
        history: session.messages.slice(0, -1),
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const finalAnswer = data.response || "No response received.";
    const thinkingText = data.thinking || "";

    // Save to session history
    session.messages.push({
      role: "assistant",
      content: finalAnswer,
      thinking: thinkingText,
    });
    saveSessions();

    // Re-render row with full markdown and copy actions
    row.remove();
    appendMessageToDOM("assistant", finalAnswer, thinkingText);

  } catch (error) {
    console.error("Chat error:", error);
    outputContainer.innerHTML = `<p style="color: #ef4444;">Failed to generate response. Please ensure SABER server is running.</p>`;
  } finally {
    state.isGenerating = false;
    elements.messageInput.focus();
  }
}

// ==========================================================================
// Event Listeners & UI Helpers
// ==========================================================================

function setupEventListeners() {
  // New chat button
  elements.newChatBtn.addEventListener("click", createNewSession);

  // Clear chat button
  elements.clearChatBtn.addEventListener("click", () => {
    const session = getCurrentSession();
    if (session) {
      session.messages = [];
      saveSessions();
      renderCurrentChat();
    }
  });

  // Toggle sidebar
  elements.toggleSidebarBtn.addEventListener("click", () => {
    elements.sidebar.classList.toggle("collapsed");
  });

  elements.mobileMenuBtn.addEventListener("click", () => {
    elements.sidebar.classList.toggle("mobile-open");
  });

  // 3-Mode Sentinel Selector
  elements.modePillBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.modePillBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.sentinelMode = btn.getAttribute("data-mode") || "1_sentinel";
    });
  });

  // Textarea input event
  elements.messageInput.addEventListener("input", () => {
    autoResizeTextarea();
    elements.sendBtn.disabled = !elements.messageInput.value.trim() || state.isGenerating;
  });

  // Keydown event (Enter to send, Shift+Enter for newline, Ctrl+K for new chat)
  elements.messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      createNewSession();
    }
  });

  // Send button click
  elements.sendBtn.addEventListener("click", () => sendMessage());

  // Prompt Cards in hero
  elements.promptCards.forEach(card => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt) {
        sendMessage(prompt);
      }
    });
  });
}

function autoResizeTextarea() {
  const input = elements.messageInput;
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Start application
document.addEventListener("DOMContentLoaded", init);
