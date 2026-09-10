import { useState, useEffect, useRef } from 'react'
import { Send, FileCode, Clock, Zap, AlertCircle } from 'lucide-react'
import { sendChatMessage, getRepositories } from '../services/api'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

function ChatPage() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [repositories, setRepositories] = useState([])
  const [selectedRepo, setSelectedRepo] = useState('')
  const [selectedLanguage, setSelectedLanguage] = useState('')
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    loadRepositories()
  }, [])

  const loadRepositories = async () => {
    try {
      const data = await getRepositories()
      setRepositories(data.repositories || [])
    } catch (error) {
      console.error('Failed to load repositories:', error)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim() || loading) return

    const userMessage = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setLoading(true)

    try {
      const requestData = {
        question: question.trim(),
        max_tokens: 512,
        temperature: 0.1,
        chat_history: messages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .slice(-6)
          .map((message) => ({
            role: message.role,
            content: buildHistoryContent(message),
          })),
      }

      if (selectedRepo) {
        requestData.repository_id = selectedRepo
        const repository = repositories.find((repo) => repo.repository_id === selectedRepo)
        if (repository) {
          requestData.repository_name = repository.repository_name
        }
      }
      if (selectedLanguage) {
        requestData.language = selectedLanguage
      }

      const response = await sendChatMessage(requestData)
      
      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        retrieval_stats: response.retrieval_stats,
        llm_stats: response.llm_stats,
        total_time_seconds: response.total_time_seconds,
      }
      
      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage = {
        role: 'error',
        content: error.response?.data?.detail || 'Failed to get response. Please try again.',
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp']

  return (
    <div className="h-[calc(100vh-12rem)] flex flex-col">
      {/* Header with filters */}
      <div className="card mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Ask Questions About Your Code</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Repository
            </label>
            <select
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              className="input-field"
            >
              <option value="">All Repositories</option>
              {repositories.map((repo) => (
                <option key={repo.repository_id} value={repo.repository_id}>
                  {repo.repository_name}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Language
            </label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="input-field"
            >
              <option value="">All Languages</option>
              {languages.map((lang) => (
                <option key={lang} value={lang}>
                  {lang.charAt(0).toUpperCase() + lang.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-4 p-4 bg-white rounded-xl shadow-sm border border-gray-200">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <FileCode className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Start a Conversation
              </h3>
              <p className="text-gray-500">
                Ask questions about your indexed codebases. I'll search through the code and provide detailed answers with source citations.
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))}
            {loading && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="card p-4">
        <div className="flex space-x-3">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your code..."
            className="input-field flex-1"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
            <span>Send</span>
          </button>
        </div>
      </form>
    </div>
  )
}

function buildHistoryContent(message) {
  if (!message.sources || message.sources.length === 0) {
    return message.content
  }

  const sourceSummary = message.sources
    .slice(0, 5)
    .map((source) => {
      const chunkName = source.chunk_name ? ` ${source.chunk_name}` : ''
      return `${source.file_path}${chunkName} lines ${source.line_start}-${source.line_end}`
    })
    .join('\n')

  return `${message.content}\n\nSources:\n${sourceSummary}`
}

function MessageBubble({ message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-2xl">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  if (message.role === 'error') {
    return (
      <div className="flex justify-start">
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-2xl">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <p>{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="bg-gray-50 border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 max-w-3xl">
        <div className="markdown-content prose prose-sm max-w-none">
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    className="code-block"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                )
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
              <FileCode className="w-4 h-4 mr-1" />
              Sources ({message.sources.length})
            </h4>
            <div className="space-y-2">
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="text-xs bg-white border border-gray-200 rounded-lg p-2 hover:border-primary-300 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-primary-700 truncate">
                        {source.file_path}
                      </p>
                      <p className="text-gray-600 mt-0.5">
                        <span className="font-medium">{source.chunk_type}</span>
                        {source.chunk_name && `: ${source.chunk_name}`}
                        <span className="text-gray-400 ml-2">
                          Lines {source.line_start}-{source.line_end}
                        </span>
                      </p>
                    </div>
                    <div className="ml-2 flex-shrink-0">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-800">
                        {(source.relevance_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        {message.llm_stats && (
          <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center space-x-4">
              <span className="flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {message.total_time_seconds?.toFixed(2)}s
              </span>
              <span className="flex items-center">
                <Zap className="w-3 h-3 mr-1" />
                {message.llm_stats.tokens_used} tokens
              </span>
            </div>
            <span className="text-gray-400">{message.llm_stats.model}</span>
          </div>
        )}
      </div>
    </div>
  )
}

function LoadingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-gray-50 border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
          <span className="text-sm text-gray-500">Thinking...</span>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
