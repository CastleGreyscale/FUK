/**
 * Tab Button Component
 */

export default function TabButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`fuk-tab ${active ? 'active' : ''}`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
