import { useStudioStore } from '../../store/studioStore'
import { TABS } from '../../constants/characters'

function ModelStatusBadge() {
  const modelStatus = useStudioStore((s) => s.modelStatus)
  const healthLoading = useStudioStore((s) => s.healthLoading)

  if (healthLoading) {
    return (
      <span className="inline-block rounded-full border border-zinc-700 bg-zinc-800/60 px-3 py-1 text-xs text-zinc-400">
        Checking model…
      </span>
    )
  }

  if (!modelStatus?.model_loaded) {
    return (
      <span className="inline-block rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs text-red-300">
        Model unavailable
      </span>
    )
  }

  const vocabLabel = modelStatus.vocab_size
    ? `${Math.round(modelStatus.vocab_size / 1000)}k vocab`
    : 'BPE'
  const device = modelStatus.device || 'cpu'

  return (
    <span className="inline-block rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300">
      GPT-2 BPE · {vocabLabel} · {device}
    </span>
  )
}

export function DegradedBanner() {
  const modelStatus = useStudioStore((s) => s.modelStatus)
  const healthLoading = useStudioStore((s) => s.healthLoading)

  if (healthLoading || modelStatus?.model_loaded) return null

  return (
    <div className="mb-6 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
      <strong>Model not loaded.</strong>{' '}
      Re-run notebooks 2–5 to produce <code className="text-amber-100">bpe_tokenizer/</code> and a new checkpoint.
      {modelStatus?.error && (
        <p className="mt-2 text-xs text-amber-300/80">{modelStatus.error}</p>
      )}
    </div>
  )
}

export default function TopBar() {
  const activeTab = useStudioStore((s) => s.activeTab)
  const setActiveTab = useStudioStore((s) => s.setActiveTab)

  return (
    <header className="mb-6 border-b border-zinc-800 pb-4">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="font-script text-2xl font-semibold text-zinc-100 sm:text-3xl">
            Shakespeare Studio
          </h1>
          <div className="mt-2">
            <ModelStatusBadge />
          </div>
        </div>
        <nav className="flex flex-wrap gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                activeTab === tab.id
                  ? 'bg-violet-600/20 text-violet-300 ring-1 ring-violet-500/40'
                  : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  )
}
