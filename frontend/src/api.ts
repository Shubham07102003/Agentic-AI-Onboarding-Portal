import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export type ChatMessage = { role: 'user' | 'assistant'; content: string; ts: number }
export type Profile = Record<string, unknown>
export type Card = {
  bank_name: string
  card_name: string
  annual_fee: string | number
  key_benefits: string
  description: string
  website: string
  card_type: string
}

export async function fetchHealth() {
  const { data } = await axios.get(`${API_BASE}/api/health`)
  return data
}

export async function fetchPrompts(): Promise<string[]> {
  const { data } = await axios.get(`${API_BASE}/api/prompts`)
  return data.prompts || []
}

export async function fetchHistory(sessionId: string) {
  const { data } = await axios.get(`${API_BASE}/api/history/${sessionId}`)
  return data as { session_id: string; chat: ChatMessage[]; profile: Profile }
}

export async function clearHistory(sessionId: string) {
  const { data } = await axios.delete(`${API_BASE}/api/history/${sessionId}`)
  return data
}

export async function sendChat(message: string, sessionId?: string) {
  const { data } = await axios.post(`${API_BASE}/api/chat`, { message, session_id: sessionId })
  return data as {
    session_id: string
    answer: string
    suggestions: string[]
    profile: Profile
    cards: Card[]
  }
}

export type StreamEvent =
  | { event: 'start'; session_id: string }
  | { event: 'delta'; text: string }
  | { event: 'end'; session_id: string; suggestions: string[]; profile: Profile; cards: Card[] }

export async function streamChat(message: string, sessionId?: string, onEvent?: (ev: StreamEvent) => void) {
  const resp = await fetch(`${API_BASE}/api/chat_stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!resp.body) return
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, idx).trim()
      buf = buf.slice(idx + 1)
      if (!line) continue
      try {
        const ev = JSON.parse(line) as StreamEvent
        onEvent?.(ev)
      } catch {}
    }
  }
}

export async function uploadCSV(file: File) {
  const form = new FormData()
  form.append('file', file, file.name)
  const { data } = await axios.post(`${API_BASE}/api/upload`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

