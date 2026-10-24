// Speech-to-text for the pronunciation check: Whisper (transformers.js) running on the device.
import { pipeline, env } from "./tf/transformers.min.js";

env.backends.onnx.wasm.wasmPaths = new URL("./tf/", import.meta.url).href;
// ?asr=<dir>&asrModel=<id> serves the model from this site instead of huggingface.co (used for local tests)
// the override is honoured on localhost only, so a shared link cannot point the model at another host
const q = /^(localhost|127\.0\.0\.1)$/.test(self.location.hostname) ? new URL(self.location.href).searchParams : new URLSearchParams();
if (q.get("asr")) { env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = q.get("asr"); }
const MODEL = q.get("asrModel") || "Xenova/whisper-tiny";
// short phrases only: a low token cap also stops the rare endless "Good. Good. Good…" loop early
const OPTS = { max_new_tokens: 48, ...(MODEL.endsWith(".en") ? {} : { language: "english", task: "transcribe" }) };

let asr = null;
const load = () => (asr = asr || pipeline("automatic-speech-recognition", MODEL, {
  device: "wasm",
  dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
  progress_callback: (p) => {
    if (p.status === "progress" && /\.onnx$/.test(p.file)) postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total });
  },
}));

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "load") { await load(); postMessage({ type: "ready" }); }
    if (m.type === "run") { const out = await (await load())(m.audio, OPTS); postMessage({ type: "text", id: m.id, text: out.text }); }
  } catch (e) {
    if (m.type === "load") asr = null;
    postMessage({ type: "error", id: m.id, error: String((e && e.message) || e) });
  }
};
