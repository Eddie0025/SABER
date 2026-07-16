// DOM Elements
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const tierDropdownToggle = document.getElementById('tier-dropdown-toggle');
const tierDropdownMenu = document.getElementById('tier-dropdown-menu');
const selectedTierInput = document.getElementById('selected-tier');
const dropdownItems = document.querySelectorAll('.dropdown-item');

// API Base URL (assuming UI is served by FastAPI or running locally)
const API_BASE = window.location.origin.includes('5173') || window.location.protocol === 'file:' 
    ? 'http://localhost:8000/api' 
    : '/api';

// State
let isWaiting = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    userInput.addEventListener('input', autoResizeTextarea);
    userInput.addEventListener('keydown', handleEnterPress);
    chatForm.addEventListener('submit', handleSubmit);
    
    // Sidebar toggle
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    // New Conversation button
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', async () => {
            try {
                await fetch(`${API_BASE}/history`, { method: 'DELETE' });
                chatHistory.innerHTML = '';
            } catch (e) {
                console.error('Failed to clear history:', e);
            }
        });
    }

    // Dropdown toggle
    tierDropdownToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        tierDropdownMenu.classList.toggle('show');
    });

    // Dropdown items selection
    dropdownItems.forEach(item => {
        item.addEventListener('click', () => {
            const val = item.getAttribute('data-value');
            selectedTierInput.value = val;
            
            // Update UI text and active states
            dropdownItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            const title = item.querySelector('.item-title').textContent;
            tierDropdownToggle.querySelector('span').textContent = title.split(' (')[0];
            
            tierDropdownMenu.classList.remove('show');
        });
    });

    // Close dropdown on click outside
    document.addEventListener('click', () => {
        tierDropdownMenu.classList.remove('show');
    });
    
    // Marked options for markdown parsing
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    // Load persisted chat history
    loadHistory();
});

// Auto-resize textarea
function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
    sendBtn.disabled = userInput.value.trim() === '' || isWaiting;
}

// Handle Enter key (Shift+Enter for new line)
function handleEnterPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (userInput.value.trim() !== '' && !isWaiting) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
}

// Handle query submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const text = userInput.value.trim();
    if (!text || isWaiting) return;
    
    // Get selected tier
    const selectedTier = parseInt(selectedTierInput.value) || 2;

    // 1. Add user message
    appendUserMessage(text);
    
    // 2. Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    // 3. Show loading
    const loadingId = appendSystemLoading();
    
    isWaiting = true;
    
    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text,
                verification_tier: selectedTier
            })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        
        // 4. Remove loading & show answer
        removeElement(loadingId);
        appendSystemResponse(data);
        
    } catch (error) {
        console.error('Query error:', error);
        removeElement(loadingId);
        appendSystemError('Communication with the Orchestrator failed. Please ensure the backend is running.');
    } finally {
        isWaiting = false;
        sendBtn.disabled = userInput.value.trim() === '';
        userInput.focus();
    }
}

// UI Helpers
function appendUserMessage(text) {
    const html = `
        <div class="message user-message">
            <div class="avatar user-avatar">U</div>
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
            </div>
        </div>
    `;
    chatHistory.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function appendSystemLoading() {
    const id = 'msg-' + Date.now();
    const html = `
        <div class="message system-message" id="${id}">
            <div class="avatar system-avatar">S</div>
            <div class="message-content">
                <div class="loader">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    chatHistory.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function appendSystemResponse(data) {
    // Parse markdown
    const formattedAnswer = marked.parse(data.answer);
    
    // Meta information (confidence, domains, tier, cycles)
    let metaHtml = '';
    if (data.status === 'complete') {
        const domains = (data.domains_activated || []).map(d => `<span style="text-transform:capitalize">${d}</span>`).join(', ');
        metaHtml = `
            <div class="response-meta">
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                    <span>Cycles: <span class="meta-val">${data.verification_cycles}</span></span>
                </div>
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    <span>Confidence: <span class="meta-val">${(data.confidence * 100).toFixed(1)}%</span></span>
                </div>
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
                    <span>Domains: <span class="meta-val">${domains}</span></span>
                </div>
            </div>
        `;
    }

    // Flags mapping
    let flagsHtml = '';
    if (data.unresolved_flags && data.unresolved_flags.length > 0) {
        flagsHtml = '<div class="flags-container">';
        data.unresolved_flags.forEach(flag => {
            flagsHtml += `
                <div class="flag-alert">
                    <div class="flag-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                        Unresolved Issue: ${flag.issue_type.replace('_', ' ')}
                    </div>
                    <div class="flag-desc">${flag.description}</div>
                </div>
            `;
        });
        flagsHtml += '</div>';
    } else if (data.status === 'clarification_needed') {
        flagsHtml = `
            <div class="flags-container">
                <div class="flag-alert" style="border-color:var(--accent-primary); background:rgba(59,130,246,0.1)">
                    <div class="flag-header" style="color:var(--accent-primary)">
                        Ambiguity Detected (Score: ${data.ambiguity_score.toFixed(2)})
                    </div>
                </div>
            </div>
        `;
    } else if (data.status === 'no_specialists') {
        flagsHtml = `
            <div class="flags-container">
                <div class="flag-alert">
                    <div class="flag-header">No Matching Domains Found</div>
                </div>
            </div>
        `;
    }

    const html = `
        <div class="message system-message">
            <div class="avatar system-avatar">S</div>
            <div class="message-content">
                ${formattedAnswer}
                ${flagsHtml}
                ${metaHtml}
            </div>
        </div>
    `;
    
    chatHistory.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function appendSystemError(text) {
    const html = `
        <div class="message system-message">
            <div class="avatar system-avatar" style="background:var(--danger)">!</div>
            <div class="message-content">
                <p style="color:var(--danger)">${text}</p>
            </div>
        </div>
    `;
    chatHistory.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Load persisted chat history from backend
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history`);
        if (!response.ok) return;
        
        const messages = await response.json();
        if (!messages || messages.length === 0) return;

        for (const msg of messages) {
            if (msg.role === 'user') {
                appendUserMessage(msg.content);
            } else if (msg.role === 'system') {
                // Reconstruct SABER response data from stored metadata
                const meta = msg.metadata || {};
                const data = {
                    answer: msg.content,
                    status: meta.status || 'complete',
                    confidence: meta.confidence || 0,
                    domains_activated: meta.domains_activated || [],
                    verification_cycles: meta.verification_cycles || 0,
                    unresolved_flags: meta.unresolved_flags || [],
                    ambiguity_score: meta.ambiguity_score || 0,
                };
                appendSystemResponse(data);
            }
        }
    } catch (e) {
        // Silently fail — history loading is non-critical
        console.log('Chat history not available:', e.message);
    }
}
