const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const CHAT_ID = 'day2-demo'

export async function uploadPDF(file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `Upload failed (${res.status})`)
  }

  return res.json()
}

export async function askQuestion(message, smartMode = false) {
  const res = await fetch(`${API}/chat?chat_id=${encodeURIComponent(CHAT_ID)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, chat_id: CHAT_ID, smart_mode: smartMode }),
  })

  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `Chat failed (${res.status})`)
  }

  return res.json()
}
