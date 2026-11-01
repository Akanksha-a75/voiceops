# ZentroMart VoiceOps (From Tokens to Transformers capstone)

Voice customer-support agent for ZentroMart: Gnani STT -> RAG over the knowledge base (Groq LLM) -> Gnani TTS.

Repo: https://github.com/Akanksha-a75/voiceops

## Setup
1. Python 3.13, then: python -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
2. Copy .env.example to .env and fill in GNANI_API_KEY and LLM_API_KEY (Groq). LLM_MODEL=openai/gpt-oss-120b
3. Set DATASET_DIR to the extracted dataset folder (default: data_local/Capstone_Project_Dataset_FTTT_2026). The dataset is not included in this repo.

## Run (one command)
    python run.py --audio_dir <folder_of_audio> --out_dir <output_folder>

For each audio file (sorted) it runs Gnani STT, the RAG agent, then Gnani TTS. Outputs in out_dir: <name>_reply.txt, <name>_reply.wav, and results.csv (transcript, decision, reply text, reply audio, document chunk IDs used).
Assumption: each audio file is one customer turn. Errors on a file never stop the batch; empty transcripts get a "please say it again" reply.

## Evaluate on practice questions
    python -m src.eval retrieval --k 5
    python -m src.eval step1
    python -m src.eval step2
Scores are saved to sample_outputs/scores.json.

## Design
- src/kb.py: chunks KB by section ID, BM25 retrieval, precedence weighting, companion-section links, catalog lookup.
- src/qa.py: Step 1 prompt-only baseline; Step 2 RAG (English query rewrite, retrieve, answer only from context, say no information otherwise).
- src/llm.py: Groq wrapper with retries.
- src/voice: Gnani STT/TTS only. No API keys in code.

## Known gaps
No multi-turn conversation agent (order verification, lookup_order, create_ticket) was built in time; Step 2 answers single questions only.