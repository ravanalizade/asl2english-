// Talks to the FastAPI backend.
const BASE = "http://localhost:8000";

export async function classifyLetter(imageDataUrl) {
  const res = await fetch(`${BASE}/letter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  if (!res.ok) throw new Error("letter request failed");
  return res.json(); // { letter, confidence }
}

export async function assembleSentence(text) {
  const res = await fetch(`${BASE}/assemble`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("assemble request failed");
  return res.json(); // { sentence } or { error }
}
