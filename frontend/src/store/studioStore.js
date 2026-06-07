import { create } from 'zustand'

const DEFAULT_CHARACTERS = ['HAMLET', 'ROMEO', 'JULIET', 'MACBETH', 'OPHELIA']

export const useStudioStore = create((set, get) => ({
  prompt: '',
  characters: [...DEFAULT_CHARACTERS],
  turns: 2,
  temperature: 0.7,
  messages: [],
  isGenerating: false,
  error: null,

  setPrompt: (prompt) => set({ prompt }),
  setCharacters: (characters) => set({ characters }),
  setTurns: (turns) => set({ turns }),
  setTemperature: (temperature) => set({ temperature }),
  setError: (error) => set({ error }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setGenerating: (isGenerating) => set({ isGenerating }),

  clearMessages: () => set({ messages: [], error: null }),

  startRoundtable: async () => {
    const { prompt, characters, turns, temperature } = get()
    if (!prompt.trim()) return

    set({ messages: [], isGenerating: true, error: null })

    try {
      const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE}/roundtable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          characters,
          turns,
          temperature,
          top_k: 40,
          max_new_tokens: 120,
        }),
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed (${response.status})`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''
        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))
            if (data.finished) {
              set({ isGenerating: false })
              return
            }
            if (data.done && data.character) {
              set((state) => ({
                messages: [...state.messages, {
                  id: crypto.randomUUID(),
                  character: data.character,
                  text: data.text,
                  turn: data.turn,
                }],
              }))
            }
          }
        }
      }
      set({ isGenerating: false })
    } catch (err) {
      set({ isGenerating: false, error: err.message })
    }
  },
}))
