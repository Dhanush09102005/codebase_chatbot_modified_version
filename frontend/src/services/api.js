import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Repository endpoints
export const indexRepository = async (data) => {
  const response = await api.post('/api/v1/repositories/index', data)
  return response.data
}

export const getRepositories = async () => {
  const response = await api.get('/api/v1/repositories')
  return response.data
}

export const deleteRepository = async (repositoryId) => {
  const response = await api.delete(`/api/v1/repositories/${repositoryId}`)
  return response.data
}

// Chat endpoint
export const sendChatMessage = async (data) => {
  const response = await api.post('/api/v1/chat', data)
  return response.data
}

// Health endpoint
export const getHealth = async () => {
  const response = await api.get('/api/v1/health')
  return response.data
}

export default api
