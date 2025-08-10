import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Upload, Send, Sparkles, RefreshCcw, Trash2, Building2, CreditCard as CardIcon, Flame, Fuel, Wallet } from 'lucide-react'
import { Card as CardType, ChatMessage, clearHistory, fetchHealth, fetchHistory, fetchPrompts, sendChat, streamChat, uploadCSV } from './api'

type ChatItem = ChatMessage & { suggestions?: string[] }

function ChatBubble({ item, onChipClick }: { item: ChatItem; onChipClick?: (s: string) => void }) {
  const isUser = item.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-2xl px-4 py-3 shadow ${isUser ? 'bg-brand-500 text-white rounded-br-sm' : 'bg-white text-neutral-900 rounded-bl-sm border border-neutral-100'}`}>
        {isUser ? (
          <div className="whitespace-pre-wrap leading-relaxed text-[15px]">{item.content}</div>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-li:my-0 prose-headings:mt-3 prose-headings:mb-1">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content}</ReactMarkdown>
          </div>
        )}
        {!!item.suggestions?.length && (
          <div className="mt-2 flex flex-wrap gap-2">
            {item.suggestions.slice(0, 6).map((s, i) => (
              <button
                type="button"
                key={i}
                onClick={() => onChipClick?.(s)}
                className="inline-block text-xs px-2 py-1 rounded-full border bg-neutral-50 text-neutral-700 border-neutral-200 hover:border-brand-400 hover:text-brand-700"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function CardResult({ card }: { card: CardType }) {
  return (
    <div className="p-4 rounded-xl border bg-white shadow-sm">
      <div className="flex items-center gap-2 text-[15px] font-semibold">
        <CardIcon className="w-4 h-4 text-brand-600" />
        {card.card_name}
        <span className="text-neutral-500 font-normal">— {card.bank_name} ({card.card_type})</span>
      </div>
      {!!card.description && <p className="mt-2 text-sm text-neutral-700">{card.description}</p>}
      {!!card.key_benefits && (
        <ul className="mt-2 pl-5 list-disc text-sm text-neutral-800 space-y-1">
          {card.key_benefits
            .split(/\n|•|;|\.|,/)
            .map((s) => s.trim())
            .filter(Boolean)
            .slice(0, 5)
            .map((s, i) => (
              <li key={i}>{s}</li>
            ))}
        </ul>
      )}
      <div className="mt-3 flex items-center gap-3 text-sm">
        {!!card.annual_fee && (
          <span className="inline-flex items-center gap-1 text-neutral-700">
            <Wallet className="w-4 h-4" /> ₹{card.annual_fee}
          </span>
        )}
        {!!card.website && (
          <a target="_blank" href={card.website} rel="noreferrer" className="text-brand-600 hover:underline">
            Apply / Details
          </a>
        )}
      </div>
    </div>
  )}

export default function App() {
  const [sessionId, setSessionId] = useState<string>('')
  const [chat, setChat] = useState<ChatItem[]>([])
  const [input, setInput] = useState('')
  const [prompts, setPrompts] = useState<string[]>([])
  const [cards, setCards] = useState<CardType[]>([])
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [health, setHealth] = useState<any>(null)

  const chatEndRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chat, cards])

  // initial load
  useEffect(() => {
    fetchPrompts().then(setPrompts).catch(() => setPrompts([]))
    fetchHealth().then(setHealth).catch(() => setHealth(null))
    const saved = localStorage.getItem('cc_session_id')
    if (saved) {
      fetchHistory(saved)
        .then((h) => {
          setSessionId(h.session_id)
          setChat(h.chat as ChatItem[])
        })
        .catch(() => {})
    }
  }, [])

  const appendAssistant = (text: string, suggestions?: string[]) => {
    const assistant: ChatItem = { role: 'assistant', content: text, ts: Math.floor(Date.now() / 1000), suggestions } as any
    setChat((c) => [...c, assistant])
  }

  const onSend = async (msg?: string) => {
    const message = (msg ?? input).trim()
    if (!message) return
    setInput('')
    const newUser: ChatItem = { role: 'user', content: message, ts: Math.floor(Date.now() / 1000) } as any
    setChat((c) => [...c, newUser])
    setLoading(true)
    setStreaming(true)
    let acc = ''
    try {
      await streamChat(message, sessionId || undefined, (ev) => {
        if (ev.event === 'start') {
          setSessionId(ev.session_id)
          localStorage.setItem('cc_session_id', ev.session_id)
          acc = ''
          setChat((c) => [...c, { role: 'assistant', content: '', ts: Math.floor(Date.now() / 1000) } as any])
        } else if (ev.event === 'delta') {
          acc += ev.text
          setChat((c) => {
            const copy = [...c]
            const last = copy[copy.length - 1]
            if (last?.role === 'assistant') {
              ;(last as any).content = acc
            }
            return copy
          })
        } else if (ev.event === 'end') {
          setCards(ev.cards || [])
          setStreaming(false)
          setLoading(false)
        }
      })
    } catch (e) {
      setStreaming(false)
      setLoading(false)
      appendAssistant('Something went wrong. Please try again.')
    }
  }

  const onClear = async () => {
    if (sessionId) {
      await clearHistory(sessionId)
    }
    localStorage.removeItem('cc_session_id')
    setSessionId('')
    setChat([])
    setCards([])
  }

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    const ok = await uploadCSV(f)
    appendAssistant(ok.message || 'Dataset uploaded & indexed.')
  }

  const headerStatus = useMemo(() => {
    if (!health) return null
    return (
      <div className="flex items-center gap-3 text-xs text-neutral-600">
        <span className={`inline-flex items-center gap-1 ${health.openai ? 'text-green-600' : 'text-red-600'}`}>
          <Sparkles className="w-4 h-4" /> OpenAI {health.openai ? 'OK' : '—'}
        </span>
        <span className={`inline-flex items-center gap-1 ${health.tavily ? 'text-green-600' : 'text-red-600'}`}>
          <Building2 className="w-4 h-4" /> Tavily {health.tavily ? 'OK' : '—'}
        </span>
        <span className="inline-flex items-center gap-1">
          <Flame className="w-4 h-4" /> {health.dataset_rows} cards
        </span>
      </div>
    )
  }, [health])

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded bg-brand-500" />
            <div>
              <div className="font-bold">Credit Card Advisor</div>
              <div className="text-xs text-neutral-500">RAG + Web Search</div>
            </div>
          </div>
          {headerStatus}
        </div>
      </header>

      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-4 py-6 grid grid-cols-1 md:grid-cols-3 gap-6">
          <section className="md:col-span-2 flex flex-col gap-4">
            <div className="rounded-2xl bg-neutral-50 border p-4">
              <div className="flex gap-2 flex-wrap">
                {prompts.map((p, idx) => (
                  <button
                    key={idx}
                    className="chip"
                    onClick={() => onSend(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 chat-container p-4 min-h-[420px] max-h-[60vh] overflow-y-auto space-y-4">
              {chat.map((m, i) => (
                <ChatBubble key={i} item={m} onChipClick={(s) => onSend(s)} />
              ))}
              <div ref={chatEndRef} />
              {(loading || streaming) && (
                <div className="text-sm text-neutral-500">Assistant is typing…</div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <div className="flex-1 flex items-center gap-2 chat-input">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      onSend()
                    }
                  }}
                  placeholder="Type your message"
                  className="flex-1 outline-none text-[15px] placeholder:text-neutral-400 bg-white text-neutral-900"
                />
                <button
                  onClick={() => onSend()}
                  className="inline-flex items-center gap-1 rounded-full bg-brand-600 text-white px-3 py-1.5 hover:bg-brand-700"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
              <button
                onClick={onClear}
                className="inline-flex items-center gap-1 rounded-full bg-neutral-200 text-neutral-800 px-3 py-1.5 hover:bg-neutral-300"
              >
                <RefreshCcw className="w-4 h-4" /> Reset
              </button>
              <label className="inline-flex items-center gap-1 rounded-full bg-neutral-800 text-white px-3 py-1.5 cursor-pointer hover:bg-black">
                <Upload className="w-4 h-4" /> Upload CSV
                <input type="file" accept=".csv" className="hidden" onChange={onUpload} />
              </label>
            </div>
          </section>

          <aside className="md:col-span-1 flex flex-col gap-3">
            <div className="rounded-2xl border bg-white p-4">
              <div className="font-semibold mb-2 flex items-center gap-2">
                <Fuel className="w-4 h-4 text-brand-600" /> Suggested cards
              </div>
              <div className="space-y-3">
                {cards.length === 0 ? (
                  <div className="text-sm text-neutral-500">No suggestions yet. Ask a question or use quick prompts.</div>
                ) : (
                  cards.map((c, i) => <CardResult key={i} card={c} />)
                )}
              </div>
            </div>

            <div className="rounded-2xl border bg-white p-4">
              <div className="font-semibold mb-2 flex items-center gap-2">
                <Trash2 className="w-4 h-4 text-brand-600" /> Safety
              </div>
              <p className="text-sm text-neutral-600">We redact sensitive PII like PAN, Aadhaar, phone and email from your messages before processing.</p>
            </div>
          </aside>
        </div>
      </main>

      <footer className="border-t">
        <div className="mx-auto max-w-6xl px-4 py-4 text-xs text-neutral-500">
          Educational, not financial advice. Verify issuer T&Cs and eligibility before applying.
        </div>
      </footer>
    </div>
  )
}


