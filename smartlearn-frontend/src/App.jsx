import { useState } from 'react'
import { uploadPDF, askQuestion } from './api.js'
import PdfUploader from './PdfUploader.jsx'
import ChatPanel from './ChatPanel.jsx'

export default function App() {
  const [upload, setUpload] = useState(null)
  const [answer, setAnswer] = useState(null)
  const [status, setStatus] = useState('')
  const [error, setError] = useState(null)

  const busy = status !== ''

  async function handleUpload(file) {
    if (!file || busy) return
    setStatus('Uploading…')
    setError(null)
    setUpload(null)
    setAnswer(null)
    try {
      const result = await uploadPDF(file)
      setUpload(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setStatus('')
    }
  }

  async function handleAsk(message) {
    if (!message || busy) return
    setStatus('Asking…')
    setError(null)
    setAnswer(null)
    try {
      const result = await askQuestion(message)
      setAnswer({ text: result.answer, citations: parseCitations(result.answer) })
    } catch (e) {
      setError(e.message)
    } finally {
      setStatus('')
    }
  }

  return (
    <main>
      <h1>SmartLearn</h1>

      <PdfUploader upload={upload} busy={busy} onUpload={handleUpload} />

      {status && <p>{status}</p>}

      {error && <p role="alert">{error}</p>}

      {upload && <ChatPanel answer={answer} busy={busy} onAsk={handleAsk} />}
    </main>
  )
}

function parseCitations(text) {
  const seen = new Set()
  const pages = []
  const regex = /\[Page (\d+)\]/g
  let match
  while ((match = regex.exec(text)) !== null) {
    const page = parseInt(match[1], 10)
    if (!seen.has(page)) {
      seen.add(page)
      pages.push(page)
    }
  }
  return pages
}
