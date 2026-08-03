import { CHAT_ID } from './api.js'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export function getDocumentFileURL(page = 1) {
  return `${API}/documents/${encodeURIComponent(CHAT_ID)}/file#page=${page}`
}

export default function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <section className="pdf-preview">
        <div className="placeholder">Upload a PDF to preview it here</div>
      </section>
    )
  }

  const url = getDocumentFileURL(activePage)

  return (
    <section className="pdf-preview">
      <div className="preview-toolbar">
        <span>Page {activePage}</span>
      </div>
      <iframe
        key={`${previewKey}-${activePage}`}
        src={url}
        title="PDF preview"
        className="preview-frame"
      />
    </section>
  )
}
