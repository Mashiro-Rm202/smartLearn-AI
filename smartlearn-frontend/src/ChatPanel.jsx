import { useState } from 'react'

export default function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Clear local messages when a new upload remounts this panel
  // (enabled flips from false→true on upload success, key={uploadKey} from App)

  async function handleSubmit(e) {
    e.preventDefault()
    const text = message.trim()
    if (!text || loading || !enabled) return

    setMessage('')
    setError(null)
    setLoading(true)
    if (onBusy) onBusy(true)

    // Append user message immediately
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])

    try {
      const { askQuestion } = await import('./api.js')
      const result = await askQuestion(text)
      const assistantMsg = {
        role: 'assistant',
        content: result.answer,
        citations: result.citations || [],
        sources: result.sources || [],
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      if (onBusy) onBusy(false)
    }
  }

  function handleCitationClick(page) {
    if (onJumpToPage) onJumpToPage(page)
  }

  return (
    <section className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="placeholder">Ask a question about the uploaded PDF</div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="message-role">{msg.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="message-content">{msg.content}</div>
            {msg.citations && msg.citations.length > 0 && (
              <div className="citation-row">
                {msg.citations.map(p => (
                  <button
                    key={p}
                    className="citation-chip"
                    onClick={() => handleCitationClick(p)}
                  >
                    Page {p}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && <div className="chat-message assistant"><div className="message-content">Thinking…</div></div>}
      </div>

      {error && <p role="alert">{error}</p>}

      <form onSubmit={handleSubmit} className="chat-form">
        <textarea
          id="message"
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Your question…"
          disabled={!enabled || disabled}
        />
        <button type="submit" disabled={!message.trim() || loading || !enabled || disabled}>
          Ask
        </button>
      </form>
    </section>
  )
}
