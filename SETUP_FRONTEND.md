# Frontend Setup Guide

This guide will help you set up and run the frontend for the Code Q&A Assistant.

## Overview

The frontend is a modern React application built with:
- **React 18** with Vite for fast development
- **Tailwind CSS** for beautiful styling
- **React Router** for navigation
- **Axios** for API communication
- **React Markdown** for rich text rendering
- **Syntax Highlighting** for code blocks

## Prerequisites

Before you begin, ensure you have:
- **Node.js 18+** and npm installed
- **Backend API** running on `http://localhost:8000`

### Check Node.js Installation

```bash
node --version  # Should be v18 or higher
npm --version   # Should be v8 or higher
```

If Node.js is not installed, download it from [nodejs.org](https://nodejs.org/)

## Installation Steps

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

This will install all required packages including:
- react, react-dom
- react-router-dom
- axios
- tailwindcss
- react-markdown
- react-syntax-highlighter
- lucide-react
- vite

### 3. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` to set your backend API URL (default is already correct):

```env
VITE_API_URL=http://localhost:8000
```

### 4. Start Development Server

```bash
npm run dev
```

The application will start at `http://localhost:3000`

You should see:
```
VITE v5.0.8  ready in 500 ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

### 5. Open in Browser

Navigate to `http://localhost:3000` in your web browser.

## Available Scripts

### Development

```bash
npm run dev
```
Starts the development server with hot module replacement (HMR).

### Build

```bash
npm run build
```
Creates an optimized production build in the `dist/` folder.

### Preview

```bash
npm run preview
```
Preview the production build locally before deploying.

### Lint

```bash
npm run lint
```
Run ESLint to check for code quality issues.

## Usage Guide

### 1. Chat Page (Home)

**Ask questions about your code:**
1. Select a repository filter (optional)
2. Select a language filter (optional)
3. Type your question in the input field
4. Click "Send" or press Enter
5. View the AI-generated answer with source citations

**Example questions:**
- "How does authentication work in this codebase?"
- "Show me all the database queries"
- "Explain the caching mechanism"
- "Where is error handling implemented?"

### 2. Repositories Page

**Index a new repository:**
1. Click "Index Repository"
2. Enter the Git URL (e.g., `https://github.com/fastapi/fastapi.git`)
3. Optionally specify a branch (default: main)
4. Click "Index Repository"
5. Wait for the indexing process to complete

**Manage repositories:**
- View all indexed repositories with metadata
- See file counts, chunk counts, and languages
- Delete repositories you no longer need

**Supported repository sources:**
- GitHub URLs: `https://github.com/user/repo.git`
- GitLab URLs: `https://gitlab.com/user/repo.git`
- Local paths: `/path/to/local/repo`

### 3. Health Page

**Monitor system status:**
- View overall system health
- Check vector store status and chunk count
- Verify embedding model configuration
- Confirm LLM provider setup
- View repository statistics
- Auto-refreshes every 30 seconds

## Features

### 🎨 Beautiful UI
- Modern, clean design with Tailwind CSS
- Responsive layout for all screen sizes
- Smooth animations and transitions
- Custom color scheme with primary blues

### 💬 Interactive Chat
- Real-time question answering
- Markdown-formatted responses
- Syntax-highlighted code blocks
- Source file citations with line numbers
- Performance statistics (response time, tokens)

### 📚 Repository Management
- Easy repository indexing
- Git clone with branch support
- View repository metadata
- Delete repositories
- File and chunk statistics

### 🔍 Advanced Filtering
- Filter by specific repository
- Filter by programming language
- Combine multiple filters

### 📊 Health Monitoring
- Component status indicators
- Repository statistics dashboard
- Real-time health checks
- Auto-refresh capability

## Troubleshooting

### Port Already in Use

If port 3000 is already in use:

1. Edit `vite.config.js`:
```js
server: {
  port: 3001,  // Change to any available port
}
```

2. Restart the dev server

### Backend Connection Issues

If you see "Failed to load repositories" or similar errors:

1. Verify backend is running:
```bash
curl http://localhost:8000/api/v1/health
```

2. Check CORS settings in backend (`app/api/app.py`)

3. Verify `.env` file has correct API URL

### Build Errors

If you encounter build errors:

1. Clear node_modules and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

2. Clear Vite cache:
```bash
rm -rf node_modules/.vite
npm run dev
```

### Styling Issues

If styles don't load correctly:

1. Ensure Tailwind CSS is configured:
```bash
npx tailwindcss init -p
```

2. Verify `index.css` imports Tailwind directives

## Production Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized build in `dist/`:
- Minified JavaScript and CSS
- Code splitting for faster loads
- Asset optimization
- Source maps for debugging

### Preview Production Build

```bash
npm run preview
```

### Deploy to Static Hosting

The `dist/` folder can be deployed to any static hosting service:

**Vercel:**
```bash
npm install -g vercel
vercel --prod
```

**Netlify:**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

**Nginx:**
```bash
# Copy dist folder to web root
cp -r dist/* /var/www/html/

# Configure nginx for React Router
location / {
  try_files $uri $uri/ /index.html;
}
```

### Environment Variables for Production

Update `.env` for production:

```env
VITE_API_URL=https://your-backend-api.com
```

Rebuild after changing environment variables:
```bash
npm run build
```

## Project Structure

```
frontend/
├── public/              # Static assets
│   └── vite.svg
├── src/
│   ├── pages/          # Page components
│   │   ├── ChatPage.jsx
│   │   ├── RepositoriesPage.jsx
│   │   └── HealthPage.jsx
│   ├── services/       # API integration
│   │   └── api.js
│   ├── App.jsx         # Main app with routing
│   ├── main.jsx        # React entry point
│   └── index.css       # Global styles & Tailwind
├── index.html          # HTML template
├── package.json        # Dependencies & scripts
├── vite.config.js      # Vite configuration
├── tailwind.config.js  # Tailwind configuration
├── postcss.config.js   # PostCSS configuration
└── .env               # Environment variables
```

## Customization

### Change Theme Colors

Edit `tailwind.config.js`:

```js
theme: {
  extend: {
    colors: {
      primary: {
        50: '#f0f9ff',
        // ... customize your colors
        900: '#0c4a6e',
      },
    },
  },
}
```

### Modify API Endpoints

Edit `src/services/api.js` to add or modify endpoints:

```js
export const customEndpoint = async (data) => {
  const response = await api.post('/api/v1/custom', data)
  return response.data
}
```

### Add New Pages

1. Create component in `src/pages/`
2. Import in `App.jsx`
3. Add route in `<Routes>` component
4. Add navigation link in `Navigation` component

## Browser Support

Tested and supported on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Tips

### Development
- Vite provides instant HMR for fast development
- React Fast Refresh preserves component state
- Only imports modules you actually use

### Production
- Automatic code splitting
- Tree shaking removes unused code
- Assets are hashed for cache busting
- Gzip compression recommended on server

## Getting Help

If you encounter issues:

1. Check console for errors (F12 in browser)
2. Verify backend is running and accessible
3. Check network tab for failed API requests
4. Review browser compatibility
5. Check GitHub issues

## Next Steps

1. **Index your first repository** on the Repositories page
2. **Ask questions** on the Chat page
3. **Monitor health** on the Health page
4. **Customize the theme** to match your brand
5. **Deploy to production** when ready

Enjoy your Code Q&A Assistant! 🚀
