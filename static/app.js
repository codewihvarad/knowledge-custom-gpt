/* ═══════════════════════════════════════════════════
   RAG Bot — app.js  (Production Frontend Logic)
   Features:
     • Upload + progress animation
     • Streaming SSE responses
     • Markdown rendering (no deps)
     • Document list management
     • Toast notifications
     • Auto-resizing textarea
     • Keyboard shortcuts
   ═══════════════════════════════════════════════════ */

const API = {
  health:    '/health',
  ingest:    '/api/ingest',
  chat:      '/api/chat',
  stream:    '/api/chat/stream',
  documents: '/api/documents',
};

/* ── State ───────────────────────────────────────── */
let isStreaming = false;
let currentStreamCtrl = null;

/* ── DOM refs ────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const messagesContainer = $('messagesContainer');
const questionInput      = $('questionInput');
const sendBtn            = $('sendBtn');
const typingIndicator    = $('typingIndicator');
const uploadZone         = $('uploadZone');
const fileInput          = $('fileInput');
const uploadProgress     = $('uploadProgress');
const progressFill       = $('progressFill');
const progressLabel      = $('progressLabel');
const docList            = $('docList');
const docCount           = $('docCount');
const streamToggle       = $('streamToggle');
const clearBtn           = $('clearBtn');
const sidebar            = $('sidebar');
const sidebarToggle      = $('sidebarToggle');
const mobileMenuBtn      = $('mobileMenuBtn');
const neo4jStatus        = $('neo4jStatus');
const ollamaStatus       = $('ollamaStatus');
const toast              = $('toast');
const toastMsg           = $('toastMsg');

/* ════════════════════ INIT ════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadDocuments();
  setupUploadZone();
  setupInputAutoResize();
  setupKeyboardShortcuts();
  setInterval(checkHealth, 30_000); // refresh status every 30s
});

/* ════════════════════ HEALTH CHECK ════════════════ */
async function checkHealth() {
  try {
    const res = await fetch(API.health);
    const data = await res.json();

    setStatus(neo4jStatus, data.neo4j === 'ok');
    setStatus(ollamaStatus, data.neo4j === 'ok'); // ollama checked indirectly
  } catch {
    setStatus(neo4jStatus, false);
    setStatus(ollamaStatus, false);
  }
}
function setStatus(el, ok) {
  el.className = `status-pill ${ok ? 'ok' : 'err'}`;
}

/* ════════════════════ DOCUMENTS ═══════════════════ */
async function loadDocuments() {
  try {
    const res  = await fetch(API.documents);
    const data = await res.json();
    renderDocList(data.documents || []);
  } catch (e) {
    console.error('Failed to load documents:', e);
  }
}

function renderDocList(docs) {
  docCount.textContent = docs.length;
  if (docs.length === 0) {
    docList.innerHTML = '<li class="doc-empty">No documents yet</li>';
    return;
  }
  docList.innerHTML = docs.map(d => `
    <li class="doc-item" data-id="${d.doc_id}">
      <span class="doc-icon">${d.filename.endsWith('.pdf') ? '📕' : '📄'}</span>
      <div class="doc-info">
        <div class="doc-name" title="${d.filename}">${d.filename}</div>
        <div class="doc-meta">${d.total_chunks} chunks</div>
      </div>
      <button class="doc-delete" onclick="deleteDocument('${d.doc_id}','${d.filename}')" title="Delete">✕</button>
    </li>
  `).join('');
}

async function deleteDocument(docId, filename) {
  if (!confirm(`Delete "${filename}"?\nThis will remove it from Neo4j permanently.`)) return;
  try {
    const res = await fetch(`${API.documents}/${docId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast(`✅ Deleted "${filename}"`, false);
      loadDocuments();
    } else {
      showToast('❌ Delete failed', true);
    }
  } catch {
    showToast('❌ Network error', true);
  }
}

/* ════════════════════ UPLOAD ══════════════════════ */
function setupUploadZone() {
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault(); uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault(); uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });
}

async function handleUpload(file) {
  const allowed = ['.pdf', '.txt', '.md'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast('❌ Only PDF, TXT, MD files are allowed', true); return;
  }

  // Show progress
  uploadProgress.classList.remove('hidden');
  uploadZone.classList.add('hidden');
  progressFill.style.width = '0%';
  progressLabel.textContent = `Ingesting "${file.name}"…`;

  // Fake progress animation while ingesting
  let fakeProgress = 0;
  const ticker = setInterval(() => {
    fakeProgress = Math.min(fakeProgress + (Math.random() * 8), 88);
    progressFill.style.width = fakeProgress + '%';
  }, 400);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(API.ingest, { method: 'POST', body: formData });
    clearInterval(ticker);

    if (res.ok) {
      const data = await res.json();
      progressFill.style.width = '100%';
      progressLabel.textContent = `✅ "${data.filename}" — ${data.total_chunks} chunks indexed`;
      showToast(`📄 Ingested "${data.filename}" (${data.total_chunks} chunks)`, false);
      loadDocuments();
      setTimeout(() => {
        uploadProgress.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        progressFill.style.width = '0%';
      }, 2500);
    } else {
      const err = await res.json();
      throw new Error(err.detail || 'Unknown error');
    }
  } catch (e) {
    clearInterval(ticker);
    progressFill.style.width = '100%';
    progressFill.style.background = 'var(--error)';
    progressLabel.textContent = `❌ Failed: ${e.message}`;
    showToast(`❌ Ingestion failed: ${e.message}`, true);
    setTimeout(() => {
      uploadProgress.classList.add('hidden');
      uploadZone.classList.remove('hidden');
      progressFill.style.background = '';
    }, 3000);
  }
  fileInput.value = '';
}

/* ════════════════════ CHAT ════════════════════════ */
async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || isStreaming) return;

  // Add user message
  appendMessage(question, 'user');
  questionInput.value = '';
  resizeTextarea();

  // Hide suggestions on first message
  $('suggestions').style.display = 'none';

  const useStream = streamToggle.checked;

  if (useStream) {
    await sendStreamingMessage(question);
  } else {
    await sendRegularMessage(question);
  }
}

async function sendRegularMessage(question) {
  setThinking(true);
  try {
    const res = await fetch(API.chat, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'API error');
    }
    const data = await res.json();
    appendBotMessage(data.answer, data.sources);
  } catch (e) {
    appendBotMessage(`❌ Error: ${e.message}`, []);
  } finally {
    setThinking(false);
  }
}

async function sendStreamingMessage(question) {
  isStreaming = true;
  sendBtn.disabled = true;
  typingIndicator.classList.remove('hidden');

  const botMsgEl = createBotMessageEl();
  const contentEl = botMsgEl.querySelector('.message-content');
  messagesContainer.appendChild(botMsgEl);
  typingIndicator.classList.add('hidden');

  let accumulated = '';
  try {
    const url = `${API.stream}?question=${encodeURIComponent(question)}`;
    const res = await fetch(url);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const token = line.slice(6);
          if (token === '[DONE]') break;
          if (token.startsWith('ERROR:')) {
            accumulated += `\n❌ ${token}`;
          } else {
            accumulated += token;
          }
          contentEl.innerHTML = renderMarkdown(accumulated) + '<span class="cursor-blink">▌</span>';
          scrollToBottom();
        }
      }
    }
    // Final render without cursor
    contentEl.innerHTML = renderMarkdown(accumulated);
  } catch (e) {
    contentEl.innerHTML = `<span style="color:var(--error)">❌ Stream error: ${e.message}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

function setThinking(thinking) {
  typingIndicator.classList.toggle('hidden', !thinking);
  sendBtn.disabled = thinking;
}

/* ════════════════════ MESSAGE RENDERING ═══════════ */
function appendMessage(text, role) {
  const el = document.createElement('div');
  el.className = `message ${role}-message`;
  el.innerHTML = `
    <div class="message-avatar ${role}-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="message-bubble">
      <div class="message-content">${role === 'user' ? escapeHtml(text) : renderMarkdown(text)}</div>
    </div>
  `;
  messagesContainer.appendChild(el);
  scrollToBottom();
}

function createBotMessageEl() {
  const el = document.createElement('div');
  el.className = 'message bot-message';
  el.innerHTML = `
    <div class="message-avatar bot-avatar">🤖</div>
    <div class="message-bubble">
      <div class="message-content"></div>
    </div>
  `;
  return el;
}

function appendBotMessage(text, sources = []) {
  const el = document.createElement('div');
  el.className = 'message bot-message';

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const tags = sources.slice(0, 5).map(s =>
      `<span class="source-tag">📄 ${s.source || 'doc'} p.${s.page ?? '?'} <span class="source-score">(${(s.score * 100).toFixed(1)}%)</span></span>`
    ).join('');
    sourcesHtml = `<div class="sources-panel"><div class="sources-title">📎 Sources</div>${tags}</div>`;
  }

  el.innerHTML = `
    <div class="message-avatar bot-avatar">🤖</div>
    <div class="message-bubble">
      <div class="message-content">${renderMarkdown(text)}</div>
      ${sourcesHtml}
    </div>
  `;
  messagesContainer.appendChild(el);
  scrollToBottom();
}

/* ════════════════════ MARKDOWN RENDERER ═══════════ */
function renderMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trimEnd()}</code></pre>`);
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');
  // Ordered list
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, m => `<ol>${m}</ol>`);
  // Unordered list
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, m => {
    if (m.includes('<ol>')) return m;
    return `<ul>${m}</ul>`;
  });
  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  // Line breaks
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;
  // Clean up p tags around block elements
  html = html.replace(/<p>(<h[1-3]>|<ul>|<ol>|<pre>|<blockquote>)/g, '$1');
  html = html.replace(/(<\/h[1-3]>|<\/ul>|<\/ol>|<\/pre>|<\/blockquote>)<\/p>/g, '$1');

  return html;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ════════════════════ INPUT & UX ══════════════════ */
function setupInputAutoResize() {
  questionInput.addEventListener('input', resizeTextarea);
}
function resizeTextarea() {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
}

function setupKeyboardShortcuts() {
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); sendMessage();
    }
  });
}

sendBtn.addEventListener('click', sendMessage);

clearBtn.addEventListener('click', () => {
  if (!confirm('Clear chat history?')) return;
  messagesContainer.innerHTML = '';
  $('suggestions').style.display = 'flex';
});

sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
mobileMenuBtn.addEventListener('click', () => sidebar.classList.toggle('mobile-open'));

function fillSuggestion(el) {
  questionInput.value = el.textContent;
  resizeTextarea();
  questionInput.focus();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });
}

/* ════════════════════ TOAST ═══════════════════════ */
let toastTimer;
function showToast(msg, isError = false) {
  toastMsg.textContent = msg;
  toast.className = `toast${isError ? ' toast-error' : ''}`;
  toast.querySelector('.toast-icon').textContent = isError ? '❌' : '✅';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
}

/* ══ Cursor blink style ══════════════════════════ */
const style = document.createElement('style');
style.textContent = `
  .cursor-blink {
    display: inline-block; animation: blink 0.8s step-end infinite;
    color: var(--accent-2); font-weight: bold;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
`;
document.head.appendChild(style);
