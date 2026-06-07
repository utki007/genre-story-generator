export const SCENE_CHARACTERS = ['ROMEO', 'JULIET', 'HAMLET', 'OPHELIA', 'MACBETH']

export const SCENE_LENGTH_PRESETS = {
  short: { turns: 1, max_new_tokens: 80 },
  medium: { turns: 2, max_new_tokens: 120 },
  long: { turns: 3, max_new_tokens: 96 },
}

export const TABS = [
  { id: 'chat', label: 'Chat' },
  { id: 'scene', label: 'Scene Generator' },
  { id: 'arena', label: 'Arena' },
  { id: 'explorer', label: 'Explorer' },
]
