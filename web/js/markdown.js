/**
 * Lightweight Markdown Parser with Code Block Syntax Highlighting & Copy Actions
 */

const MarkdownRenderer = {
  escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  },

  render(markdown) {
    if (!markdown) return "";

    let text = markdown;

    // 1. Code blocks with language detection and syntax highlighting
    text = text.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = (lang || "text").toLowerCase();
      const codeId = "code_" + Math.random().toString(36).substring(2, 9);
      
      let highlightedCode = this.escapeHtml(code.trim());
      if (window.hljs && window.hljs.getLanguage(language)) {
        try {
          highlightedCode = window.hljs.highlight(code.trim(), { language }).value;
        } catch (e) {
          // fallback to escaped
        }
      }

      return `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${language}</span>
            <button class="copy-code-btn" onclick="MarkdownRenderer.copyCode('${codeId}')">
              <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              <span>Copy</span>
            </button>
          </div>
          <pre><code id="${codeId}" class="hljs language-${language}">${highlightedCode}</code></pre>
        </div>
      `;
    });

    // 2. Inline Code
    text = text.replace(/`([^`]+)`/g, (match, code) => {
      return `<code>${this.escapeHtml(code)}</code>`;
    });

    // 3. Headers
    text = text.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    text = text.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    text = text.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // 4. Bold & Italic
    text = text.replace(/\*\*\*(.*?)\*\*\*/g, "<strong><em>$1</em></strong>");
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // 5. Unordered Lists
    text = text.replace(/^\s*[-*]\s+(.*)$/gim, "<li>$1</li>");
    text = text.replace(/(<li>.*<\/li>)/gim, "<ul>$1</ul>");
    text = text.replace(/<\/ul>\s*<ul>/gim, ""); // Merge consecutive lists

    // 6. Ordered Lists
    text = text.replace(/^\s*\d+\.\s+(.*)$/gim, "<ol-item>$1</ol-item>");
    text = text.replace(/(<ol-item>.*<\/ol-item>)/gim, "<ol>$1</ol>");
    text = text.replace(/<ol-item>/gim, "<li>").replace(/<\/ol-item>/gim, "</li>");
    text = text.replace(/<\/ol>\s*<ol>/gim, "");

    // 7. Paragraphs
    const blocks = text.split(/\n\n+/);
    text = blocks
      .map(b => {
        b = b.trim();
        if (
          b.startsWith("<h1>") ||
          b.startsWith("<h2>") ||
          b.startsWith("<h3>") ||
          b.startsWith("<ul>") ||
          b.startsWith("<ol>") ||
          b.startsWith("<div class=\"code-block-wrapper\"")
        ) {
          return b;
        }
        return `<p>${b.replace(/\n/g, "<br>")}</p>`;
      })
      .join("\n");

    return text;
  },

  copyCode(codeId) {
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;
    const textToCopy = codeEl.innerText || codeEl.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      // Find button and show 'Copied!'
      const btn = codeEl.closest(".code-block-wrapper").querySelector(".copy-code-btn span");
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => {
          btn.textContent = orig;
        }, 2000);
      }
    });
  }
};
