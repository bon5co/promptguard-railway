# promptguard-railway

Prompt-injection and safety classifier API for [Railway](https://railway.com), CPU only. Two open-weights models behind one authenticated HTTP API:

- [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (Apache-2.0): SAFE / INJECTION score for text headed into an LLM prompt. English only; its model card says it does not detect jailbreak attacks.
- [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) (Apache-2.0): Safe / Controversial / Unsafe label plus categories, 119 languages.

- `POST /scan` `{"text": "..." | ["a", "b"], "threshold": 0.5, "safety": true}` -> `{"results": [{"injection": {"label", "score", "tokens", "windows"}, "safety": {"label", "categories"} | null, "flagged": bool}]}`. Texts over 512 tokens are scored in overlapping windows and the highest score is reported.
- `GET /health` (503 until both models are loaded and warm, then 200), `GET /v1/models`.
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: `API_KEY` (required), `INJECTION_THRESHOLD` (0.5), `MAX_TEXT_CHARS` (20000), `MAX_BATCH` (32), `TORCH_THREADS` (from the cgroup quota).

Weights are baked into the image, pinned to one Hugging Face commit each and verified by sha256 at build. Text is held in memory for one request and never logged or stored. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`.

Image: `ghcr.io/bon5co/promptguard-railway`.
