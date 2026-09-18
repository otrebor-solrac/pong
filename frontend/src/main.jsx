import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[RootErrorBoundary] Error caught:', error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '30px', color: '#f87171', background: '#0f172a', fontFamily: 'monospace', minHeight: '100vh' }}>
          <h2 style={{ color: '#ef4444', marginBottom: '16px' }}>Application Render Error</h2>
          <pre style={{ background: '#1e293b', padding: '15px', borderRadius: '8px', overflowX: 'auto', color: '#fca5a5', marginBottom: '16px' }}>
            {this.state.error?.toString()}
          </pre>
          <pre style={{ background: '#1e293b', padding: '15px', borderRadius: '8px', overflowX: 'auto', color: '#94a3b8', fontSize: '12px' }}>
            {this.state.errorInfo?.componentStack || this.state.error?.stack}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{ marginTop: '20px', padding: '10px 20px', background: '#38bdf8', color: '#0f172a', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600 }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>,
)

