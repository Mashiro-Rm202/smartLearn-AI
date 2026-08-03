import { useState } from 'react'
import { uploadPDF } from './api.js'
import PdfUploader from './PdfUploader.jsx'
import ChatPanel from './ChatPanel.jsx'
import PdfPreview from './PdfPreview.jsx'

export default function App() {
  const [upload, setUpload] = useState(null)
  const [activePage, setActivePage] = useState(1)
  const [uploadKey, setUploadKey] = useState(0)
  const [status, setStatus] = useState('')
  const [error, setError] = useState(null)

  const busy = status !== ''

  async function handleUpload(file) {
    if (!file || busy) return
    setStatus('Uploading…')
    setError(null)
    setUpload(null)
    try {
      const result = await uploadPDF(file)
      setUpload(result)
      setActivePage(1)
      setUploadKey(k => k + 1)  // remount chat panel
    } catch (e) {
      setError(e.message)
    } finally {
      setStatus('')
    }
  }

  function handleJumpToPage(page) {
    setActivePage(page)
  }

  return (
    <main className="day3-layout">
      <h1 className="app-title">SmartLearn</h1>

      <PdfUploader upload={upload} busy={busy} onUpload={handleUpload} />

      <div className="workspace">
        <PdfPreview upload={upload} activePage={activePage} previewKey={uploadKey} />
        <ChatPanel
          key={uploadKey}
          enabled={!!upload}
          onBusy={(b) => setStatus(b ? 'Asking…' : '')}
          disabled={busy}
          onJumpToPage={handleJumpToPage}
        />
      </div>

      {status && <p className="status-line">{status}</p>}
      {error && <p role="alert">{error}</p>}
    </main>
  )
}
