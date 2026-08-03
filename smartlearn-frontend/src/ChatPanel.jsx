import { useEffect, useRef, useState } from 'react'

export default function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

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
      <div className="chat-header">
        <div>
          <span className="panel-eyebrow">AI workspace</span>
          <strong>Document chat</strong>
        </div>
        <span className={`connection-state ${enabled ? 'ready' : ''}`}>
          <span aria-hidden="true" />
          {enabled ? 'Ready' : 'Waiting for PDF'}
        </span>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <div className="chat-empty-mark" aria-hidden="true">AI</div>
            <strong>{enabled ? 'Your document is ready' : 'Start with a PDF'}</strong>
            <p>{enabled ? 'Ask a question and get an answer grounded in the document.' : 'Upload a document to begin asking questions.'}</p>
          </div>
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

        {loading && (
          <div className="chat-message assistant is-loading">
            <div className="message-role">Assistant</div>
            <div className="thinking-dots" aria-label="Thinking">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {error && <p className="chat-error" role="alert">{error}</p>}

      <form onSubmit={handleSubmit} className="chat-form">
        <textarea
          id="message"
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Your question…"
          disabled={!enabled || disabled}
          rows={2}
        />
        <button type="submit" disabled={!message.trim() || loading || !enabled || disabled}>
          Ask
        </button>
      </form>
    </section>
  )
}
