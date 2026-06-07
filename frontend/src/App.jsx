import { useRef, useEffect } from 'react'
import { useStudioStore } from './store/studioStore'

const CHARACTER_COLORS = {
  HAMLET: { bg: 'bg-violet-500/15', border: 'border-violet-500/40', text: 'text-violet-300', badge: 'bg-violet-500/20 text-violet-300' },
  ROMEO: { bg: 'bg-rose-500/15', border: 'border-rose-500/40', text: 'text-rose-300', badge: 'bg-rose-500/20 text-rose-300' },
  JULIET: { bg: 'bg-pink-500/15', border: 'border-pink-500/40', text: 'text-pink-300', badge: 'bg-pink-500/20 text-pink-300' },
  MACBETH: { bg: 'bg-red-500/15', border: 'border-red-500/40', text: 'text-red-300', badge: 'bg-red-500/20 text-red-300' },
  OPHELIA: { bg: 'bg-sky-500/15', border: 'border-sky-500/40', text: 'text-sky-300', badge: 'bg-sky-500/20 text-sky-300' },
  MERCUTIO: { bg: 'bg-amber-500/15', border: 'border-amber-500/40', text: 'text-amber-300', badge: 'bg-amber-500/20 text-amber-300' },
  IAGO: { bg: 'bg-emerald-500/15', border: 'border-emerald-500/40', text: 'text-emerald-300', badge: 'bg-emerald-500/20 text-emerald-300' },
  OTHELLO: { bg: 'bg-orange-500/15', border: 'border-orange-500/40', text: 'text-orange-300', badge: 'bg-orange-500/20 text-orange-300' },
  LEAR: { bg: 'bg-yellow-500/15', border: 'border-yellow-500/40', text: 'text-yellow-300', badge: 'bg-yellow-500/20 text-yellow-300' },
  HORATIO: { bg: 'bg-teal-500/15', border: 'border-teal-500/40', text: 'text-teal-300', badge: 'bg-teal-500/20 text-teal-300' },
}

const DEFAULT_COLOR = { bg: 'bg-zinc-500/15', border: 'border-zinc-500/40', text: 'text-zinc-300', badge: 'bg-zinc-500/20 text-zinc-300' }

function getColor(character) {
  return CHARACTER_COLORS[character] || DEFAULT_COLOR
}

const ALL_CHARACTERS = [
  'HAMLET', 'OPHELIA', 'HORATIO',
  'ROMEO', 'JULIET', 'MERCUTIO',
  'MACBETH', 'IAGO', 'OTHELLO', 'LEAR',
]

function TypingIndicator({ character }) {
  const color = getColor(character)
  return (
    <div className={`animate-fade-in-up rounded-xl border ${color.border} ${color.bg} p-4`}>
      <span className={`text-xs font-semibold uppercase tracking-wider ${color.text}`}>
        {character}
      </span>
      <div className="mt-2 flex gap-1">
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-400"></span>
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-400"></span>
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-400"></span>
      </div>
    </div>
  )
}

function MessageCard({ message }) {
  const color = getColor(message.character)
  return (
    <div className={`animate-fade-in-up rounded-xl border ${color.border} ${color.bg} p-4`}>
      <div className="mb-2 flex items-center gap-2">
        <span className={`rounded-md px-2 py-0.5 text-xs font-semibold uppercase tracking-wider ${color.badge}`}>
          {message.character}
        </span>
        {message.turn > 0 && (
          <span className="text-[10px] text-zinc-500">Round {message.turn + 1}</span>
        )}
      </div>
      <p className="font-script text-lg leading-relaxed text-zinc-200 whitespace-pre-wrap">
        {message.text}
      </p>
    </div>
  )
}

function PromptInput() {
  const prompt = useStudioStore((s) => s.prompt)
  const setPrompt = useStudioStore((s) => s.setPrompt)
  const isGenerating = useStudioStore((s) => s.isGenerating)
  const startRoundtable = useStudioStore((s) => s.startRoundtable)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!isGenerating && prompt.trim()) {
      startRoundtable()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a topic or scene for the characters to discuss..."
          disabled={isGenerating}
          rows={3}
          className="w-full resize-none rounded-xl border border-zinc-700/50 bg-zinc-800/80 px-5 py-4 pr-24 text-base text-zinc-100 placeholder-zinc-500 outline-none transition focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/30 disabled:opacity-50"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e)
            }
          }}
        />
        <button
          type="submit"
          disabled={isGenerating || !prompt.trim()}
          className="absolute right-3 bottom-3 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isGenerating ? 'Generating...' : 'Discuss'}
        </button>
      </div>
    </form>
  )
}

function CharacterPicker() {
  const characters = useStudioStore((s) => s.characters)
  const setCharacters = useStudioStore((s) => s.setCharacters)
  const isGenerating = useStudioStore((s) => s.isGenerating)

  const toggle = (char) => {
    if (isGenerating) return
    if (characters.includes(char)) {
      if (characters.length > 2) {
        setCharacters(characters.filter((c) => c !== char))
      }
    } else if (characters.length < 6) {
      setCharacters([...characters, char])
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      {ALL_CHARACTERS.map((char) => {
        const selected = characters.includes(char)
        const color = getColor(char)
        return (
          <button
            key={char}
            type="button"
            onClick={() => toggle(char)}
            disabled={isGenerating}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
              selected
                ? `${color.border} ${color.bg} ${color.text}`
                : 'border-zinc-700/50 bg-zinc-800/40 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'
            }`}
          >
            {char}
          </button>
        )
      })}
    </div>
  )
}

function Settings() {
  const turns = useStudioStore((s) => s.turns)
  const temperature = useStudioStore((s) => s.temperature)
  const setTurns = useStudioStore((s) => s.setTurns)
  const setTemperature = useStudioStore((s) => s.setTemperature)
  const isGenerating = useStudioStore((s) => s.isGenerating)

  return (
    <div className="flex items-center gap-6 text-xs text-zinc-400">
      <label className="flex items-center gap-2">
        <span>Rounds:</span>
        <select
          value={turns}
          onChange={(e) => setTurns(Number(e.target.value))}
          disabled={isGenerating}
          className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-zinc-200 outline-none disabled:opacity-50"
        >
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={3}>3</option>
        </select>
      </label>
      <label className="flex items-center gap-2">
        <span>Creativity:</span>
        <input
          type="range"
          min="0.3"
          max="1.2"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
          disabled={isGenerating}
          className="h-1 w-20 cursor-pointer appearance-none rounded-full bg-zinc-700 disabled:opacity-50"
        />
        <span className="w-6 text-zinc-300">{temperature.toFixed(1)}</span>
      </label>
    </div>
  )
}

function Conversation() {
  const messages = useStudioStore((s) => s.messages)
  const isGenerating = useStudioStore((s) => s.isGenerating)
  const characters = useStudioStore((s) => s.characters)
  const error = useStudioStore((s) => s.error)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isGenerating])

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
        <strong>Error:</strong> {error}
      </div>
    )
  }

  if (messages.length === 0 && !isGenerating) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 py-20 text-center">
        <div className="text-4xl opacity-30">&#127917;</div>
        <p className="max-w-md text-sm text-zinc-500">
          Pick your characters above, type a topic, and watch them have a conversation in Shakespearean style.
        </p>
      </div>
    )
  }

  const nextCharIdx = messages.length % characters.length
  const nextChar = characters[nextCharIdx]

  return (
    <div className="flex flex-col gap-3">
      {messages.map((msg) => (
        <MessageCard key={msg.id} message={msg} />
      ))}
      {isGenerating && <TypingIndicator character={nextChar} />}
      <div ref={endRef} />
    </div>
  )
}

export default function App() {
  const clearMessages = useStudioStore((s) => s.clearMessages)
  const messages = useStudioStore((s) => s.messages)
  const isGenerating = useStudioStore((s) => s.isGenerating)

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col px-4 py-8">
      {/* Header */}
      <header className="mb-8 text-center">
        <h1 className="font-script text-3xl font-semibold text-zinc-100">
          Shakespeare Roundtable
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          Characters discuss your topic in their own voice
        </p>
      </header>

      {/* Controls */}
      <section className="mb-6 space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 backdrop-blur">
        <CharacterPicker />
        <PromptInput />
        <div className="flex items-center justify-between">
          <Settings />
          {messages.length > 0 && !isGenerating && (
            <button
              onClick={clearMessages}
              className="rounded-md px-3 py-1 text-xs text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-300"
            >
              Clear
            </button>
          )}
        </div>
      </section>

      {/* Conversation */}
      <section className="flex-1">
        <Conversation />
      </section>
    </div>
  )
}
