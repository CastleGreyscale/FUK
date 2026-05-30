import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/index.css'

// Keep slider fill (--fill CSS var) in sync with value for all .fuk-slider elements
function syncSliderFill(el) {
  const min = parseFloat(el.min) || 0;
  const max = parseFloat(el.max) || 100;
  const val = parseFloat(el.value) || 0;
  const pct = max === min ? 0 : ((val - min) / (max - min)) * 100;
  el.style.setProperty('--fill', `${pct}%`);
}

document.addEventListener('input', (e) => {
  if (e.target.matches('.fuk-slider, .fuk-range, .gallery-zoom-slider')) syncSliderFill(e.target);
});

new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.matches('.fuk-slider, .fuk-range, .gallery-zoom-slider')) syncSliderFill(node);
      node.querySelectorAll('.fuk-slider, .fuk-range, .gallery-zoom-slider').forEach(syncSliderFill);
    }
  }
}).observe(document.documentElement, { childList: true, subtree: true });

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
