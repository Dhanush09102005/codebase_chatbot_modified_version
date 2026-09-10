# Code Q&A Assistant - Complete Setup Guide

This document provides a complete guide for setting up both the backend and frontend of the Code Q&A Assistant.

## Project Overview

**Code Q&A Assistant** is a RAG-based (Retrieval-Augmented Generation) system that helps you understand codebases by:
- Indexing Git repositories using AST-based parsing
- Creating semantic code chunks (functions, classes, methods)
- Generating embeddings for vector search
- Answering questions using hybrid retrieval (vector + keyword search)
- Citing exact file locations and line numbers

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│  - Chat Interface  - Repository Manager  - Health View  │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  - API Routes  - RAG Chain  - Retrieval  - LLM Client  │
└─────────────────────┬───────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌──────────┐   ┌───────────┐   ┌──────────┐
│ ChromaDB │   │ tree-sitter│   │ LLM API  │
│ (Vector) │   │  (Parser)  │   │ (Groq/   │
│  Store   │   │            │   │  OpenAI) │
└──────────┘   └───────────┘   └──────────┘
```

## Quick Start

### Prerequisites

**Backend:**
- Python 3.10+
- pip or uv

**Frontend:**
- Node.js 18+
- npm

### Installation

#### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Edit .env with your settings (LLM provider, API keys, etc.)
nano .env

# Start the backend
python main.py
```

Backend will run on `http://localhost:8000`

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Configure environment
cp .env.example .env

# Start the frontend
npm run dev
```

Frontend will run on `http://localhost:3000`

### First Steps

1. **Open the frontend** at `http://localhost:3000`
2. **Navigate to Repositories** page
3. **Index a repository** (try `https://github.com/psf/requests.git`)
4. **Go to Chat** page and ask questions about the code
5. **View Health** page to monitor system status

## Detailed Setup Guides

### Backend Configuration

See the main README.md for detailed backend setup including:
- LLM provider configuration (HuggingFace, Groq, OpenAI, Ollama)
- Embedding model selection
- Retrieval parameters
- ChromaDB setup
- Repository management

Key environment variables:

```env
# LLM Configuration
LLM_PROVIDER=groq                                    # or huggingface, openai, ollama
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
GROQ_API_KEY=gsk_...                                 # Your API key

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Retrieval Configuration
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
HYBRID_ALPHA=0.5
```

### Frontend Configuration

See `SETUP_FRONTEND.md` for detailed frontend setup including:
- Dependencies installation
- Environment configuration
- Development workflow
- Production build
- Deployment options

Key environment variable:

```env
VITE_API_URL=http://localhost:8000
```

## Features

### Backend Features
✅ **AST-based Code Parsing** - Uses tree-sitter for 40+ languages  
✅ **Semantic Chunking** - Preserves code structure (functions, classes)  
✅ **Hybrid Search** - Vector similarity + BM25 keyword matching  
✅ **Multi-LLM Support** - HuggingFace, Groq, OpenAI, Ollama  
✅ **Reranking** - Improves retrieval quality  
✅ **Source Citations** - Exact file paths and line numbers  
✅ **Persistent Storage** - ChromaDB for embeddings  
✅ **Git Integration** - Clone and cache repositories  

### Frontend Features
✅ **Interactive Chat** - Ask questions with AI-powered answers  
✅ **Repository Management** - Index, view, delete repositories  
✅ **Advanced Filtering** - By repository and language  
✅ **Source Citations** - Clickable file references  
✅ **Health Monitoring** - System status dashboard  
✅ **Modern UI** - Beautiful, responsive design  
✅ **Syntax Highlighting** - Code blocks with proper formatting  
✅ **Performance Stats** - Response time and token usage  

## API Documentation

### Index Repository
```http
POST /api/v1/repositories/index
Content-Type: application/json

{
  "repository_url": "https://github.com/user/repo.git",
  "branch": "main",
  "force_refresh": false
}
```

### Send Chat Message
```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "How does authentication work?",
  "repository_name": "fastapi",
  "language": "python",
  "max_tokens": 512,
  "temperature": 0.1
}
```

### List Repositories
```http
GET /api/v1/repositories
```

### Delete Repository
```http
DELETE /api/v1/repositories/{repository_id}
```

### Get Health
```http
GET /api/v1/health
```

## Development Workflow

### Running Both Services

**Terminal 1 - Backend:**
```bash
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Making Changes

**Backend changes:**
1. Edit Python files in `app/`
2. Server auto-reloads with uvicorn
3. Test changes immediately

**Frontend changes:**
1. Edit React files in `frontend/src/`
2. Vite provides instant HMR
3. See changes in browser immediately

### Testing the Integration

1. **Test health endpoint:**
```bash
curl http://localhost:8000/api/v1/health
```

2. **Test from frontend:**
- Open `http://localhost:3000/health`
- Should show green status indicators

3. **Test indexing:**
- Use Repositories page to index a repo
- Monitor backend logs for progress

4. **Test chat:**
- Ask a question on Chat page
- Verify response includes sources

## Production Deployment

### Backend Deployment

**Docker:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

**Systemd Service:**
```ini
[Unit]
Description=Code Q&A Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/code-qa
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Frontend Deployment

**Build:**
```bash
cd frontend
npm run build
```

**Deploy to Nginx:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /var/www/code-qa/frontend/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Backend Issues

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**ChromaDB errors:**
```bash
rm -rf data/chroma_db
# Reindex repositories
```

**LLM API errors:**
- Verify API keys in `.env`
- Check rate limits
- Try different LLM provider

### Frontend Issues

**Dependencies:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Backend connection:**
- Verify backend is running on port 8000
- Check CORS settings in backend
- Verify `.env` has correct API URL

**Build issues:**
```bash
rm -rf node_modules/.vite dist
npm run build
```

### Common Issues

**Q: "Failed to load repositories"**  
A: Backend not running or CORS misconfigured

**Q: "No repositories yet"**  
A: Index a repository first on Repositories page

**Q: "Indexing takes too long"**  
A: Large repos can take several minutes; this is normal

**Q: "Chat gives empty responses"**  
A: Check LLM API key and quota; try different model

## Project Structure

```
code-qa-assistant/
├── app/                      # Backend application
│   ├── api/                  # API routes and models
│   ├── core/                 # Configuration and logging
│   ├── ingestion/            # Repository and file scanning
│   └── rag/                  # RAG pipeline components
├── frontend/                 # Frontend application
│   ├── src/
│   │   ├── pages/           # React page components
│   │   ├── services/        # API integration
│   │   ├── App.jsx          # Main app component
│   │   └── index.css        # Global styles
│   └── package.json
├── data/                     # Data storage
│   ├── chroma_db/           # Vector database
│   └── repos/               # Cached repositories
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
├── main.py                   # Backend entry point
├── .env.example             # Backend environment template
├── README.md                # Backend documentation
└── SETUP_FRONTEND.md        # Frontend setup guide
```

## Tech Stack Summary

**Backend:**
- FastAPI (web framework)
- tree-sitter (code parsing)
- sentence-transformers (embeddings)
- ChromaDB (vector store)
- GitPython (git operations)
- Groq/OpenAI/HuggingFace/Ollama (LLM)

**Frontend:**
- React 18 (UI library)
- Vite (build tool)
- Tailwind CSS (styling)
- React Router (navigation)
- Axios (HTTP client)
- React Markdown (rendering)
- React Syntax Highlighter (code blocks)

## Performance Tips

### Backend
- Use GPU for embeddings if available (`EMBEDDING_DEVICE=cuda`)
- Adjust `RETRIEVAL_TOP_K` based on needs (lower = faster)
- Use Groq for fastest LLM inference
- Enable reranking for better quality

### Frontend
- Use production build for deployment
- Enable gzip compression on server
- Use CDN for static assets
- Implement response caching

## Security Considerations

### Backend
- Never commit `.env` file with API keys
- Use environment variables for secrets
- Implement rate limiting for production
- Add authentication for multi-user deployments
- Sanitize repository URLs

### Frontend
- HTTPS only in production
- Validate user inputs
- Implement CSRF protection
- Set appropriate CORS policies
- Use secure cookies for sessions

## Next Steps

1. **Explore the codebase** - Review implementation details
2. **Index your repositories** - Start with small repos first
3. **Tune parameters** - Experiment with retrieval settings
4. **Add features** - Extend with your own improvements
5. **Deploy to production** - Share with your team

## Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com
- **React Docs:** https://react.dev
- **ChromaDB Docs:** https://docs.trychroma.com
- **tree-sitter:** https://tree-sitter.github.io
- **Tailwind CSS:** https://tailwindcss.com

## Support

For issues or questions:
1. Check existing documentation
2. Review error messages and logs
3. Test with minimal examples
4. Create detailed issue reports

## License

This project demonstrates a complete RAG-based code understanding system with modern frontend and backend technologies.

---

**Happy coding! 🚀**
