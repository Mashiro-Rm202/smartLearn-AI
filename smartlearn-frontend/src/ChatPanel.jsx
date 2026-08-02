import { useState } from 'react'

export default function ChatPanel({ answer, busy, onAsk }) {
  const [message, setMessage] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (!message.trim() || busy) return
    onAsk(message.trim())
  }

  return (
    <section>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="message">Your question</label>
          <textarea
            id="message"
            value={message}
            onChange={e => setMessage(e.target.value)}
          />
        </div>
        <button type="submit" disabled={!message.trim() || busy}>
          Ask
        </button>
      </form>

      {answer && (
        <div>
          <p>{answer.text}</p>
          {answer.citations.length > 0 && (
            <div>
              {answer.citations.map(p => (
                <span key={p}>Page {p}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
