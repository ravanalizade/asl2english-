import { useRef, useState } from "react";
import { useHandTracking } from "./useHandTracking";
import { classifyLetter, assembleSentence } from "./api";

export default function App() {
  const videoRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Camera off");
  const [letters, setLetters] = useState(""); // raw spelled letters
  const [sentence, setSentence] = useState(""); // Gemini output
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleCapture(dataUrl) {
    try {
      const res = await classifyLetter(dataUrl);
      if (res.letter) setLetters((prev) => prev + res.letter);
    } catch {
      setError("Could not reach the letter service. Is the backend running?");
    }
  }

  useHandTracking({
    videoRef,
    enabled: cameraOn,
    onCapture: handleCapture,
    onStatus: setStatus,
  });

  const handleDelete = () => setLetters((p) => p.slice(0, -1));
  const handleDeleteAll = () => {
    setLetters("");
    setSentence("");
    setError("");
  };

  async function handleSend() {
    if (!letters) return;
    setBusy(true);
    setError("");
    try {
      const res = await assembleSentence(letters);
      if (res.error) setError(res.error);
      else setSentence(res.sentence);
    } catch {
      setError("Could not reach the sentence service. Is the backend running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <div className="card">
        <header>
          <h1>Fingerspelling</h1>
          <p className="subtitle">Sign letter by letter — we'll shape the sentence.</p>
        </header>

        <div className="stage">
          <video
            ref={videoRef}
            className={`video ${cameraOn ? "" : "hidden"}`}
            playsInline
            muted
          />
          {!cameraOn && (
            <div className="placeholder">
              <span>Camera is off</span>
            </div>
          )}
          {cameraOn && <div className="status-pill">{status}</div>}
        </div>

        {!cameraOn ? (
          <button className="btn primary" onClick={() => setCameraOn(true)}>
            Open camera
          </button>
        ) : (
          <div className="controls">
            <button className="btn" onClick={handleDelete} disabled={!letters}>
              Delete
            </button>
            <button className="btn" onClick={handleDeleteAll} disabled={!letters}>
              Delete all
            </button>
            <button
              className="btn primary inline"
              onClick={handleSend}
              disabled={busy || !letters}
            >
              {busy ? "Thinking…" : "Send"}
            </button>
          </div>
        )}

        <div className="output">
          <label>Letters</label>
          <div className="letters-bar">
            {letters || <span className="muted">—</span>}
          </div>

          <label>Sentence</label>
          <div className="sentence-bar">
            {sentence || <span className="muted">—</span>}
          </div>

          {error && <div className="error">{error}</div>}
        </div>
      </div>

      <footer className="foot">Single-signer demo · right hand</footer>
    </div>
  );
}
