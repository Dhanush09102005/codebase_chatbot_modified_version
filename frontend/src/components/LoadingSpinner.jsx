import { Loader } from 'lucide-react'

function LoadingSpinner({ size = 'md', text = null }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }

  return (
    <div className="flex flex-col items-center justify-center space-y-3">
      <Loader className={`${sizeClasses[size]} animate-spin text-primary-600`} />
      {text && <p className="text-sm text-gray-500">{text}</p>}
    </div>
  )
}

export default LoadingSpinner
