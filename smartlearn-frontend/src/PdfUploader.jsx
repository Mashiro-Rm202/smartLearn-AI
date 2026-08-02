import { useState } from 'react'

export default function PdfUploader({ upload, busy, onUpload }) {
  const [file, setFile] = useState(null)

  function handleSubmit(e) {
    e.preventDefault()
    if (!file || busy) return
    onUpload(file)
  }

  return (
    <section>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="pdf-file">PDF file</label>
          <input
            id="pdf-file"
            type="file"
            accept="application/pdf"
            onChange={e => setFile(e.target.files[0])}
          />
        </div>
        <button type="submit" disabled={!file || busy}>
          Upload
        </button>
      </form>

      {upload && (
        <div>
          <p>Uploaded: {upload.filename}</p>
          <p>
            {upload.pages} page{upload.pages !== 1 ? 's' : ''},{' '}
            {upload.characters.toLocaleString()} character{upload.characters !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </section>
  )
}
