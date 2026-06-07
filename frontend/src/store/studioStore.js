import { create } from 'zustand'
import { getHealth } from '../api/client'
import { SCENE_LENGTH_PRESETS } from '../constants/characters'

export const useStudioStore = create((set, get) => ({
  activeTab: 'scene',
  prompt: '',
  characters: ['ROMEO', 'JULIET'],
  sceneLength: 'medium',
  temperature: 0.8,
  messages: [],
  isGenerating: false,
  error: null,
  modelStatus: null,
  healthLoading: true,

  setActiveTab: (activeTab) => set({ activeTab }),
  setPrompt: (prompt) => set({ prompt }),
  setCharacters: (characters) => set({ characters }),
  setSceneLength: (sceneLength) => set({ sceneLength }),
  setTemperature: (temperature) => set({ temperature }),
  setError: (error) => set({ error }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setGenerating: (isGenerating) => set({ isGenerating }),

  clearMessages: () => set({ messages: [], error: null }),

  fetchHealth: async () => {
    set({ healthLoading: true })
    try {
      const { data } = await getHealth()
      set({ modelStatus: data, healthLoading: false })
    } catch (err) {
      set({
        modelStatus: {
          status: 'degraded',
          model_loaded: false,
          error: err.message || 'Backend unreachable',
        },
        healthLoading: false,
      })
    }
  },

  generateScene: async () => {
    const { prompt, characters, sceneLength, temperature, modelStatus } = get()
    if (!prompt.trim()) return
    if (modelStatus && !modelStatus.model_loaded) {
      set({ error: modelStatus.error || 'Model not loaded. Re-run notebooks 2–5.' })
      return
    }

    const preset = SCENE_LENGTH_PRESETS[sceneLength] || SCENE_LENGTH_PRESETS.medium

    set({ messages: [], isGenerating: true, error: null })

    try {
      const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE}/roundtable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          characters,
          turns: preset.turns,
          temperature,
          top_k: 40,
          max_new_tokens: preset.max_new_tokens,
        }),
      })

      if (!response.ok) {
        let detail = `Request failed (${response.status})`
        try {
          const payload = await response.json()
          detail = payload.detail || detail
        } catch {
          const text = await response.text()
          if (text) detail = text
        }
        throw new Error(detail)
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
