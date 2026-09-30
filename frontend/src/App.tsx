import { useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 100 * 1024 * 1024;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");

  function selectFile(candidate?: File) {
    setError("");
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setFile(null);
      setError("Choose a JPEG, PNG, or WebP image. Video analysis will be added after the image workflow.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setFile(null);
      setError("The selected image exceeds the 100 MB upload limit.");
      return;
    }
    setFile(candidate);
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="DeepTrace home"><span className="brand-mark">D</span>DeepTrace</a>
        <span className="status"><i /> Research prototype</span>
      </header>
      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">VISUAL MEDIA SCREENING</p>
        <h1 id="page-title">See beyond <span>the surface.</span></h1>
        <p className="intro">Examine visual media for signs of manipulation with an AI-assisted screening model.</p>
      </section>
      <section className="upload-card" aria-label="Image upload">
        <div className="upload-icon" aria-hidden="true">↑</div>
        <h2>Start with an image</h2>
        <p className="muted">Image analysis is the first development milestone. Video support will follow.</p>
        <label className="file-button">
          Choose image
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => selectFile(event.target.files?.[0])} />
        </label>
        <p className="file-guidance">JPEG, PNG, or WebP · Up to 100 MB</p>
        {file && <div className="selected-file" role="status"><span>{file.name}</span><button type="button" onClick={() => setFile(null)} aria-label="Remove selected image">Remove</button></div>}
        {error && <p className="error" role="alert">{error}</p>}
        <p className="notice">DeepTrace is a research prototype. Its output will be a model score, not proof of authenticity.</p>
      </section>
      <footer>DEEPTRACE <span>·</span> AI-BASED DEEPFAKE DETECTION</footer>
    </main>
  );
}
