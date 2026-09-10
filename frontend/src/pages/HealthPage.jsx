import { useState, useEffect } from 'react'
import { Activity, CheckCircle, AlertCircle, Database, Cpu, Zap, Loader } from 'lucide-react'
import { getHealth, getRepositories } from '../services/api'

function HealthPage() {
  const [health, setHealth] = useState(null)
  const [repositories, setRepositories] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadHealthData()
    const interval = setInterval(loadHealthData, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  const loadHealthData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [healthData, repoData] = await Promise.all([
        getHealth(),
        getRepositories(),
      ])
      setHealth(healthData)
      setRepositories(repoData.repositories || [])
    } catch (err) {
      console.error('Failed to load health data:', err)
      setError('Failed to load system health information')
    } finally {
      setLoading(false)
    }
  }

  if (loading && !health) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    )
  }

  if (error && !health) {
    return (
      <div className="card">
        <div className="flex items-center space-x-3 text-red-600">
          <AlertCircle className="w-6 h-6" />
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const totalChunks = repositories.reduce((sum, repo) => sum + (repo.chunks_count || 0), 0)
  const totalFiles = repositories.reduce((sum, repo) => sum + (repo.files_count || 0), 0)

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">System Health</h2>
          <p className="text-gray-500 mt-1">Monitor your Code Q&A Assistant</p>
        </div>
        <button
          onClick={loadHealthData}
          className="btn-secondary flex items-center space-x-2"
        >
          <Activity className="w-5 h-5" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Overall Status */}
      <div className="card mb-6">
        <div className="flex items-center space-x-4">
          {health?.status === 'healthy' ? (
            <>
              <div className="p-3 bg-green-100 rounded-full">
                <CheckCircle className="w-8 h-8 text-green-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900">System Healthy</h3>
                <p className="text-gray-500">All components are operational</p>
              </div>
            </>
          ) : (
            <>
              <div className="p-3 bg-yellow-100 rounded-full">
                <AlertCircle className="w-8 h-8 text-yellow-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900">System Degraded</h3>
                <p className="text-gray-500">Some components may be unavailable</p>
              </div>
            </>
          )}
        </div>
        {health?.timestamp && (
          <p className="text-sm text-gray-500 mt-4">
            Last checked: {new Date(health.timestamp).toLocaleString()}
          </p>
        )}
      </div>

      {/* Component Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {/* Vector Store */}
        <StatusCard
          title="Vector Store"
          icon={Database}
          status={health?.vector_store?.status}
          details={[
            { label: 'Type', value: health?.vector_store?.type || 'N/A' },
            { label: 'Chunks Indexed', value: health?.vector_store?.chunks_indexed?.toLocaleString() || '0' },
          ]}
        />

        {/* Embedding Model */}
        <StatusCard
          title="Embedding Model"
          icon={Cpu}
          status={health?.embedding_model?.status}
          details={[
            { label: 'Model', value: health?.embedding_model?.model || 'N/A' },
            { label: 'Dimension', value: health?.embedding_model?.dimension || 'N/A' },
          ]}
        />

        {/* LLM Provider */}
        <StatusCard
          title="LLM Provider"
          icon={Zap}
          status={health?.llm_provider?.status}
          details={[
            { label: 'Provider', value: health?.llm_provider?.provider || 'N/A' },
            { label: 'Model', value: health?.llm_provider?.model || 'N/A' },
          ]}
        />
      </div>

      {/* Statistics */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Repository Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-gradient-to-br from-primary-50 to-primary-100 rounded-lg">
            <p className="text-4xl font-bold text-primary-700 mb-2">
              {repositories.length}
            </p>
            <p className="text-sm text-primary-900 font-medium">Indexed Repositories</p>
          </div>
          <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
            <p className="text-4xl font-bold text-purple-700 mb-2">
              {totalFiles.toLocaleString()}
            </p>
            <p className="text-sm text-purple-900 font-medium">Code Files</p>
          </div>
          <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
            <p className="text-4xl font-bold text-green-700 mb-2">
              {totalChunks.toLocaleString()}
            </p>
            <p className="text-sm text-green-900 font-medium">Searchable Chunks</p>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ title, icon: Icon, status, details }) {
  const isHealthy = status === 'healthy' || status === 'configured'
  
  return (
    <div className="card">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${isHealthy ? 'bg-green-100' : 'bg-gray-100'}`}>
            <Icon className={`w-6 h-6 ${isHealthy ? 'text-green-600' : 'text-gray-400'}`} />
          </div>
          <div>
            <h4 className="font-semibold text-gray-900">{title}</h4>
            <p className={`text-xs ${isHealthy ? 'text-green-600' : 'text-gray-500'}`}>
              {isHealthy ? 'Operational' : 'Unknown'}
            </p>
          </div>
        </div>
        {isHealthy && <CheckCircle className="w-5 h-5 text-green-500" />}
      </div>
      
      <div className="space-y-2">
        {details.map((detail, idx) => (
          <div key={idx} className="flex justify-between text-sm">
            <span className="text-gray-600">{detail.label}:</span>
            <span className="font-medium text-gray-900 text-right truncate ml-2 max-w-[200px]" title={detail.value}>
              {detail.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default HealthPage
