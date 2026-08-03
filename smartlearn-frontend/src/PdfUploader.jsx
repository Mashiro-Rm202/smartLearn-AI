import { useState } from 'react'

export default function PdfUploader({ upload, busy, onUpload }) {
  const [file, setFile] = useState(null)

  function handleSubmit(e) {
    e.preventDefault()
    if (!file || busy) return
    onUpload(file)
  }

  return (
    <section className="upload-panel" aria-label="PDF upload">
      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="file-field">
          <label htmlFor="pdf-file">Choose a PDF</label>
          <input
            id="pdf-file"
            type="file"
            accept="application/pdf"
            onChange={e => setFile(e.target.files[0])}
          />
        </div>
        <button className="upload-button" type="submit" disabled={!file || busy}>
          {busy ? 'Working…' : 'Upload'}
        </button>
      </form>

      {upload && (
        <div className="upload-summary">
          <div className="file-badge" aria-hidden="true">PDF</div>
          <div className="file-details">
            <p className="file-name" title={upload.filename}>{upload.filename}</p>
            <p className="file-stats">
              {upload.pages} page{upload.pages !== 1 ? 's' : ''}
              <span aria-hidden="true">·</span>
              {upload.characters.toLocaleString()} character{upload.characters !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
      )}
    </section>
  )
}
