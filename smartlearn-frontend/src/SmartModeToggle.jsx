export default function SmartModeToggle({ enabled, disabled, onChange }) {
  return (
    <button
      type="button"
      className={`smart-mode-toggle ${enabled ? 'active' : ''}`}
      aria-pressed={enabled}
      disabled={disabled}
      onClick={() => onChange(!enabled)}
      title="Use AI to understand intent and standardize retrieval queries"
    >
      <span className="smart-mode-icon" aria-hidden="true">✦</span>
      Smart
    </button>
  )
}
