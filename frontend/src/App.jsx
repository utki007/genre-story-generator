import { useEffect } from 'react'
import { useStudioStore } from './store/studioStore'
import TopBar, { DegradedBanner } from './components/layout/TopBar'
import SceneGenerator from './components/tabs/SceneGenerator'
import ChatTab from './components/tabs/ChatTab'
import Arena from './components/tabs/Arena'
import ModelExplorer from './components/tabs/ModelExplorer'

const TAB_CONTENT = {
  chat: ChatTab,
  scene: SceneGenerator,
  arena: Arena,
  explorer: ModelExplorer,
}

export default function App() {
  const activeTab = useStudioStore((s) => s.activeTab)
  const fetchHealth = useStudioStore((s) => s.fetchHealth)

  useEffect(() => {
    fetchHealth()
  }, [fetchHealth])

  const TabContent = TAB_CONTENT[activeTab] || SceneGenerator

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-6 sm:px-6 sm:py-8">
      <TopBar />
      <DegradedBanner />
      <main>
        <TabContent />
      </main>
    </div>
  )
}
