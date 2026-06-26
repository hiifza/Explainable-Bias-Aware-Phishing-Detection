import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './styles/globals.css'

// Apply persisted theme before first paint
const stored = localStorage.getItem('phishguard-prefs')
if (stored) {
  try {
    const { state } = JSON.parse(stored)
    if (state?.theme) {
      document.documentElement.setAttribute('data-theme', state.theme)
    }
  } catch {}
}
if (!document.documentElement.getAttribute('data-theme')) {
  document.documentElement.setAttribute('data-theme', 'dark')
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
