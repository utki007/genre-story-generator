import { useStudioStore } from '../../store/studioStore'
import { SCENE_CHARACTERS } from '../../constants/characters'

const SCENE_LENGTHS = [
  { id: 'short', label: 'Short' },
  { id: 'medium', label: 'Medium' },
  { id: 'long', label: 'Long' },
]

export default function SceneSetupPanel() {
  const prompt = useStudioStore((s) => s.prompt)
  const setPrompt = useStudioStore((s) => s.setPrompt)
  const characters = useStudioStore((s) => s.characters)
  const setCharacters = useStudioStore((s) => s.setCharacters)
  const sceneLength = useStudioStore((s) => s.sceneLength)
  const setSceneLength = useStudioStore((s) => s.setSceneLength)
  const temperature = useStudioStore((s) => s.temperature)
  const setTemperature = useStudioStore((s) => s.setTemperature)
  const isGenerating = useStudioStore((s) => s.isGenerating)
  const modelStatus = useStudioStore((s) => s.modelStatus)
  const healthLoading = useStudioStore((s) => s.healthLoading)
  const generateScene = useStudioStore((s) => s.generateScene)
  const clearMessages = useStudioStore((s) => s.clearMessages)
  const messages = useStudioStore((s) => s.messages)

  const modelReady = modelStatus?.model_loaded && !healthLoading
  const disabled = isGenerating || !modelReady

  const toggleCharacter = (char) => {
    if (isGenerating) return
    if (characters.includes(char)) {
      if (characters.length > 2) {
        setCharacters(characters.filter((c) => c !== char))
      }
    } else if (characters.length < 5) {
      setCharacters([...characters, char])
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!isGenerating && prompt.trim() && modelReady) {
      generateScene()
    }
  }

  return (
    <aside className="flex flex-col gap-5 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 backdrop-blur">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
        Scene setup
      </h2>

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <label className="flex flex-col gap-2">
          <span className="text-xs font-medium text-zinc-400">Scene Description</span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Romeo confesses his love to Juliet."
            disabled={disabled}
            rows={4}
            className="w-full resize-none rounded-xl border border-zinc-700/50 bg-zinc-800/80 px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/30 disabled:opacity-50"
          />
        </label>

        <fieldset className="flex flex-col gap-2">
          <legend className="text-xs font-medium text-zinc-400">Characters</legend>
          <div className="flex flex-col gap-1.5">
            {SCENE_CHARACTERS.map((char) => {
              const selected = characters.includes(char)
              return (
                <label
                  key={char}
                  className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm transition ${
                    selected
                      ? 'border-violet-500/40 bg-violet-500/10 text-violet-200'
                      : 'border-zinc-700/50 bg-zinc-800/40 text-zinc-400 hover:border-zinc-600'
                  } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => toggleCharacter(char)}
                    disabled={disabled}
                    className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500/30"
                  />
                  <span className="capitalize">{char.charAt(0) + char.slice(1).toLowerCase()}</span>
                </label>
              )
            })}
          </div>
        </fieldset>

        <fieldset className="flex flex-col gap-2">
          <legend className="text-xs font-medium text-zinc-400">Scene Length</legend>
          <div className="flex gap-2">
            {SCENE_LENGTHS.map(({ id, label }) => (
              <label
                key={id}
                className={`flex flex-1 cursor-pointer items-center justify-center rounded-lg border px-2 py-2 text-xs font-medium transition ${
                  sceneLength === id
                    ? 'border-violet-500/40 bg-violet-500/10 text-violet-200'
                    : 'border-zinc-700/50 bg-zinc-800/40 text-zinc-400 hover:border-zinc-600'
                } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
              >
                <input
                  type="radio"
                  name="sceneLength"
                  value={id}
                  checked={sceneLength === id}
                  onChange={() => setSceneLength(id)}
                  disabled={disabled}
                  className="sr-only"
                />
                {label}
              </label>
            ))}
          </div>
        </fieldset>

        <label className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-400">Creativity</span>
            <span className="text-xs text-zinc-300">{temperature.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min="0.3"
            max="1.2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            disabled={disabled}
            className="h-1 w-full cursor-pointer appearance-none rounded-full bg-zinc-700 disabled:opacity-50"
          />
        </label>

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={disabled || !prompt.trim()}
            className="flex-1 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isGenerating ? 'Generating…' : 'Generate Scene'}
          </button>
          {messages.length > 0 && !isGenerating && (
            <button
              type="button"
              onClick={clearMessages}
              className="rounded-lg border border-zinc-700 px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200"
            >
              Clear
            </button>
          )}
        </div>
      </form>
    </aside>
  )
}
