import SceneSetupPanel from '../scene/SceneSetupPanel'
import ScriptPanel from '../scene/ScriptPanel'

export default function SceneGenerator() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
      <SceneSetupPanel />
      <ScriptPanel />
    </div>
  )
}
