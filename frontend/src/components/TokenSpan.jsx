function logprobToColor(lp) {
  const surprise = Math.min(Math.abs(lp) / 5, 1)
  if (surprise > 0.6) return 'bg-amber-100/10 text-amber-200'
  if (surprise > 0.3) return 'bg-teal-500/10 text-teal-200'
  return ''
}

export function TokenSpan({ token, logprob }) {
  const colorClass = logprob != null ? logprobToColor(logprob) : ''
  return (
    <span className={`rounded px-0.5 ${colorClass}`}>
      {token}
    </span>
  )
}

export function TokenText({ tokens }) {
  if (!tokens?.length) return null
  return (
    <span className="whitespace-pre-wrap">
      {tokens.map((t, i) => (
        <TokenSpan key={i} token={t.token} logprob={t.logprob} />
      ))}
    </span>
  )
}
