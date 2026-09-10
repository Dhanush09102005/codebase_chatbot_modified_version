import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { MessageSquare, Database, Activity, Github } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import RepositoriesPage from './pages/RepositoriesPage'
import HealthPage from './pages/HealthPage'

function Navigation() {
  const location = useLocation()
  
  const navItems = [
    { path: '/', icon: MessageSquare, label: 'Chat' },
    { path: '/repositories', icon: Database, label: 'Repositories' },
    { path: '/health', icon: Activity, label: 'Health' },
  ]
  
  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <Github className="w-8 h-8 text-primary-600" />
            <div>
              <h1 className="text-xl font-bold text-gray-900">Code Q&A Assistant</h1>
              <p className="text-xs text-gray-500">AI-powered codebase understanding</p>
            </div>
          </div>
          
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-50 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/repositories" element={<RepositoriesPage />} />
            <Route path="/health" element={<HealthPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
