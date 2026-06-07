import { useEffect, useRef } from 'react'
import { useStudioStore } from '../../store/studioStore'

function ScriptLine({ message }) {
  return (
    <div className="animate-fade-in-up">
      <div className="font-semibold uppercase tracking-wide text-zinc-300">
        {message.character}:
      </div>
      <p className="mt-1 font-script text-lg leading-relaxed text-zinc-200 whitespace-pre-wrap">
        {message.text}
      </p>
    </div>
  )
}

function ScriptTypingIndicator({ character }) {
  return (
    <div className="animate-fade-in-up">
      <div className="font-semibold uppercase tracking-wide text-zinc-400">
        {character}:
      </div>
      <div className="mt-2 flex gap-1">
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-500" />
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-500" />
        <span className="typing-dot h-2 w-2 rounded-full bg-zinc-500" />
      </div>
    </div>
  )
}

export default function ScriptPanel() {
  const messages = useStudioStore((s) => s.messages)
  const isGenerating = useStudioStore((s) => s.isGenerating)
  const characters = useStudioStore((s) => s.characters)
  const error = useStudioStore((s) => s.error)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isGenerating])

  const nextCharIdx = messages.length % characters.length
  const nextChar = characters[nextCharIdx]

  return (
    <section className="flex min-h-[420px] flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-zinc-400">
        Generated play
      </h2>

      {error && (
        <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="flex-1 space-y-6 overflow-y-auto">
        {messages.length === 0 && !isGenerating && !error && (
          <p className="py-16 text-center text-sm text-zinc-500">
            Describe a scene and click Generate Scene.
          </p>
        )}

        {messages.map((msg) => (
          <ScriptLine key={msg.id} message={msg} />
        ))}

        {isGenerating && <ScriptTypingIndicator character={nextChar} />}
        <div ref={endRef} />
      </div>
    </section>
  )
}
