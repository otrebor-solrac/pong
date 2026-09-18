import React from 'react';

export default function ControlsPanel({
  player1Mode,
  onPlayer1ModeChange,
  aiMode,
  onModeChange,
  gameRunning,
  setGameRunning,
  ballSpeedMultiplier,
  setBallSpeedMultiplier,
  apiUrl,
  onRestartMatch,
  recordFailures = false,
  onToggleRecordFailures,
  failureCount = 0,
  onClearFailures
}) {
  const [reloadStatus, setReloadStatus] = React.useState('');

  const handleReloadWeights = async () => {
    try {
      setReloadStatus('Reloading...');
      const res = await fetch(`${apiUrl}/api/reload_models`, { method: 'POST' });
      if (res.ok) {
        window.dispatchEvent(new CustomEvent('pong:reload_models'));
        setReloadStatus('Weights Updated!');
        setTimeout(() => setReloadStatus(''), 2500);
      } else {
        setReloadStatus('Error');
        setTimeout(() => setReloadStatus(''), 2500);
      }
    } catch (err) {
      setReloadStatus('Error');
      setTimeout(() => setReloadStatus(''), 2500);
    }
  };

  const handlePlayer2ModeClick = async (newMode) => {
    onModeChange(newMode);
    try {
      const endpoint = apiUrl ? `${apiUrl}/api/mode` : '/api/mode';
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode })
      });
    } catch (err) {
      // Offline fallback
    }
  };

  return (
    <div className="controls-panel">
      <div className="panel-header">
        <h2 className="panel-title">Match settings</h2>
      </div>

      {/* Match Actions (Start / Pause and Restart) */}
      <div className="control-group">
        <div className="match-actions-grid">
          <button
            className={`game-toggle-btn ${gameRunning ? 'pause' : 'run'}`}
            onClick={() => setGameRunning(!gameRunning)}
          >
            {gameRunning ? 'Pause' : 'Start'}
          </button>
          <button
            className="game-restart-btn"
            onClick={onRestartMatch}
            title="Reset scores and center ball"
          >
            Restart
          </button>
        </div>
      </div>

      {/* Failure Dataset Recording */}
      <div className="control-group failure-recording-group">
        <div className="recording-header">
          <label className="group-label">Failure dataset</label>
        </div>
        <div className="failure-switch-row">
          <label
            className="switch-wrapper"
            title="Automatically log missed shots into data/failed_shots.json for clinic retraining"
          >
            <input
              type="checkbox"
              className="switch-input"
              checked={recordFailures}
              onChange={(e) => onToggleRecordFailures && onToggleRecordFailures(e.target.checked)}
            />
            <span className="switch-track">
              <span className="switch-thumb" />
            </span>
            <span className={`switch-status-label ${recordFailures ? 'active' : ''}`}>
              {recordFailures ? 'Recording on' : 'Recording off'}
            </span>
          </label>
          <button
            className="clear-dataset-btn"
            onClick={onClearFailures}
            title="Clear all recorded failure shots in data/failed_shots.json"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Player 1 Mode Selector (Left Paddle) */}
      <div className="control-group">
        <label className="group-label">Player 1 (left)</label>
        <div className="segmented-grid">
          <button
            className={`mode-btn-compact ${player1Mode === 'human' ? 'active' : ''}`}
            onClick={() => onPlayer1ModeChange('human')}
          >
            Human
          </button>
          <button
            className={`mode-btn-compact ${player1Mode === 'heuristic' ? 'active' : ''}`}
            onClick={() => onPlayer1ModeChange('heuristic')}
          >
            Heuristic
          </button>
          <button
            className={`mode-btn-compact ${player1Mode === 'dqn' ? 'active' : ''}`}
            onClick={() => onPlayer1ModeChange('dqn')}
          >
            DQN
          </button>
          <button
            className={`mode-btn-compact ${player1Mode === 'q_learning' ? 'active' : ''}`}
            onClick={() => onPlayer1ModeChange('q_learning')}
          >
            Q-Learning
          </button>
          <button
            className={`mode-btn-compact ${player1Mode === 'random' ? 'active' : ''}`}
            onClick={() => onPlayer1ModeChange('random')}
          >
            Random
          </button>
        </div>
      </div>

      {/* Player 2 Mode Selector (Right Paddle / AI) */}
      <div className="control-group">
        <label className="group-label">Player 2 (right / ai)</label>
        <div className="segmented-grid">
          <button
            className={`mode-btn-compact ${aiMode === 'dqn' ? 'active' : ''}`}
            onClick={() => handlePlayer2ModeClick('dqn')}
          >
            DQN
          </button>
          <button
            className={`mode-btn-compact ${aiMode === 'q_learning' ? 'active' : ''}`}
            onClick={() => handlePlayer2ModeClick('q_learning')}
          >
            Q-Learning
          </button>
          <button
            className={`mode-btn-compact ${aiMode === 'heuristic' ? 'active' : ''}`}
            onClick={() => handlePlayer2ModeClick('heuristic')}
          >
            Heuristic
          </button>
          <button
            className={`mode-btn-compact ${aiMode === 'random' ? 'active' : ''}`}
            onClick={() => handlePlayer2ModeClick('random')}
          >
            Random
          </button>
        </div>
      </div>

      {/* Model Weights Hot-Reload */}
      <div className="control-group">
        <div className="recording-header">
          <div className="label-with-info">
            <label className="group-label">Model weights</label>
            <div className="info-badge-wrap">
              <span className="info-icon" tabIndex="0">i</span>
              <div className="info-tooltip-content">
                Hot-reload newly trained weights from disk (.onnx and .npy) into the live running session without restarting the server.
              </div>
            </div>
          </div>
          {reloadStatus && <span className="weights-status-badge">{reloadStatus}</span>}
        </div>
        <button
          className="reload-weights-btn"
          onClick={handleReloadWeights}
          title="Hot-reload ONNX and Q-table weights from disk without restarting"
        >
          Reload weights
        </button>
      </div>

      {/* Ball Speed Slider */}
      <div className="control-group">
        <div className="slider-header">
          <label className="group-label">Ball speed multiplier</label>
          <span className="slider-value">{ballSpeedMultiplier.toFixed(1)}x</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="2.5"
          step="0.1"
          value={ballSpeedMultiplier}
          onChange={(e) => setBallSpeedMultiplier(parseFloat(e.target.value))}
          className="speed-slider"
        />
      </div>
    </div>
  );
}
