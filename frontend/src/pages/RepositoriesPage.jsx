import { useState, useEffect } from 'react'
import { Plus, Trash2, GitBranch, Calendar, FileCode, Loader, AlertCircle, CheckCircle, Folder } from 'lucide-react'
import { indexRepository, getRepositories, deleteRepository } from '../services/api'

function RepositoriesPage() {
  const [repositories, setRepositories] = useState([])
  const [loading, setLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)

  useEffect(() => {
    loadRepositories()
  }, [])

  const loadRepositories = async () => {
    setLoading(true)
    try {
      const data = await getRepositories()
      setRepositories(data.repositories || [])
    } catch (error) {
      console.error('Failed to load repositories:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (repositoryId, repositoryName) => {
    if (!confirm(`Are you sure you want to delete ${repositoryName}?`)) return

    try {
      await deleteRepository(repositoryId)
      await loadRepositories()
    } catch (error) {
      console.error('Failed to delete repository:', error)
      alert('Failed to delete repository. Please try again.')
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Repositories</h2>
          <p className="text-gray-500 mt-1">Manage your indexed codebases</p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Index Repository</span>
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <Loader className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      ) : repositories.length === 0 ? (
        <div className="card text-center py-12">
          <Folder className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Repositories Yet</h3>
          <p className="text-gray-500 mb-6">
            Get started by indexing your first repository
          </p>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary inline-flex items-center space-x-2"
          >
            <Plus className="w-5 h-5" />
            <span>Index Repository</span>
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {repositories.map((repo) => (
            <div key={repo.repository_id} className="card hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    <h3 className="text-xl font-semibold text-gray-900">
                      {repo.repository_name}
                    </h3>
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                      Indexed
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-4 font-mono break-all">
                    {repo.repository_url || 'Local repository'}
                  </p>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <FileCode className="w-4 h-4 text-primary-600" />
                      <span>{repo.files_count || 0} files</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <Folder className="w-4 h-4 text-primary-600" />
                      <span>{repo.chunks_count || 0} chunks</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <GitBranch className="w-4 h-4 text-primary-600" />
                      <span>{repo.metadata?.branch || 'main'}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <Calendar className="w-4 h-4 text-primary-600" />
                      <span>{new Date(repo.indexed_at).toLocaleDateString()}</span>
                    </div>
                  </div>

                  {repo.languages && repo.languages.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">Languages:</span>
                      <div className="flex flex-wrap gap-2">
                        {repo.languages.map((lang) => (
                          <span
                            key={lang}
                            className="px-2 py-1 bg-primary-50 text-primary-700 text-xs font-medium rounded"
                          >
                            {lang}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleDelete(repo.repository_id, repo.repository_name)}
                  className="btn-danger ml-4 flex items-center space-x-2"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Delete</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showAddModal && (
        <AddRepositoryModal
          onClose={() => setShowAddModal(false)}
          onSuccess={loadRepositories}
        />
      )}
    </div>
  )
}

function AddRepositoryModal({ onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    repository_url: '',
    branch: '',
    force_refresh: false,
  })
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setStatus('Cloning repository...')

    try {
      const requestData = {
        repository_url: formData.repository_url.trim(),
      }

      if (formData.branch.trim()) {
        requestData.branch = formData.branch.trim()
      }

      if (formData.force_refresh) {
        requestData.force_refresh = true
      }

      setStatus('Scanning files...')
      const response = await indexRepository(requestData)
      
      setStatus('Indexing complete!')
      setTimeout(() => {
        onSuccess()
        onClose()
      }, 1500)
    } catch (err) {
      console.error('Indexing error:', err)
      setError(err.response?.data?.detail || 'Failed to index repository. Please try again.')
      setStatus(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full p-6">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Index New Repository</h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Repository URL or Local Path *
            </label>
            <input
              type="text"
              value={formData.repository_url}
              onChange={(e) => setFormData({ ...formData, repository_url: e.target.value })}
              placeholder="https://github.com/username/repo.git or /path/to/local/repo"
              className="input-field"
              required
              disabled={loading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Supports GitHub, GitLab, or local file paths
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Branch (optional)
            </label>
            <input
              type="text"
              value={formData.branch}
              onChange={(e) => setFormData({ ...formData, branch: e.target.value })}
              placeholder="main"
              className="input-field"
              disabled={loading}
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="force_refresh"
              checked={formData.force_refresh}
              onChange={(e) => setFormData({ ...formData, force_refresh: e.target.checked })}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              disabled={loading}
            />
            <label htmlFor="force_refresh" className="ml-2 text-sm text-gray-700">
              Force re-index (if already indexed)
            </label>
          </div>

          {status && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center space-x-2">
              <Loader className="w-5 h-5 text-blue-600 animate-spin" />
              <span className="text-blue-800 text-sm font-medium">{status}</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start space-x-2">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <span className="text-red-800 text-sm">{error}</span>
            </div>
          )}

          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? 'Indexing...' : 'Index Repository'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default RepositoriesPage
