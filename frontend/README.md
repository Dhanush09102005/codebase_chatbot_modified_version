# Code Q&A Assistant - Frontend

A modern React-based frontend for the Code Q&A Assistant, featuring repository management, interactive chat, and system health monitoring.

## Features

- 🤖 **Interactive Chat** - Ask questions about your codebase with AI-powered responses
- 📚 **Repository Management** - Index, view, and manage Git repositories
- 🔍 **Advanced Filtering** - Filter by repository and programming language
- 📊 **Source Citations** - View exact file locations and code references
- 💚 **Health Dashboard** - Monitor system components and statistics
- 🎨 **Modern UI** - Built with React, Tailwind CSS, and Lucide icons

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **React Syntax Highlighter** - Code highlighting
- **Lucide React** - Icon library

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Configure the API URL in `.env`:
```env
VITE_API_URL=http://localhost:8000
```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Production Build

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── pages/           # Page components
│   │   ├── ChatPage.jsx
│   │   ├── RepositoriesPage.jsx
│   │   └── HealthPage.jsx
│   ├── services/        # API services
│   │   └── api.js
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Pages

### Chat Page
- Ask questions about your codebase
- View AI-generated answers with source citations
- Filter by repository and programming language
- See performance statistics (response time, tokens used)

### Repositories Page
- Index new repositories from Git URLs or local paths
- View all indexed repositories with metadata
- Delete repositories
- See file counts, chunk counts, and languages

### Health Page
- Monitor system health status
- View vector store, embedding model, and LLM provider status
- Track repository statistics
- Auto-refresh every 30 seconds

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`:

- `POST /api/v1/repositories/index` - Index a repository
- `GET /api/v1/repositories` - List repositories
- `DELETE /api/v1/repositories/{id}` - Delete repository
- `POST /api/v1/chat` - Send chat message
- `GET /api/v1/health` - Get system health

## Customization

### Theme Colors

Edit `tailwind.config.js` to customize the color scheme:

```js
colors: {
  primary: {
    // Customize primary colors
  }
}
```

### API Base URL

Edit `.env` to change the backend API URL:

```env
VITE_API_URL=http://your-backend-url:8000
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is part of the Code Q&A Assistant system.
