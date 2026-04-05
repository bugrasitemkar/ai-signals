import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import SignalDetailPage from './pages/SignalDetailPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/signals/:signalId" element={<SignalDetailPage />} />
    </Routes>
  )
}
