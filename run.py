"""Batch runner: folder of audio -> Gnani STT -> RAG agent -> Gnani TTS.
python run.py --audio_dir <folder> --out_dir <folder>
Each audio file is treated as one customer turn."""
import argparse, csv, glob, os, re, sys, time
from dotenv import load_dotenv
load_dotenv()

FALLBACK = "Sorry, I didn't catch that. Could you say it again?"
EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")


def clean_for_speech(t):
    t = re.sub(r"\u20b9\s?(\d[\d,]*)", r"\1 rupees", t)
    t = re.sub(r"\bRs\.?\s?(\d[\d,]*)", r"\1 rupees", t)
    return t


def transcribe(stt, path):
    for lang in ("en-IN", "hi-IN"):
        for _ in range(2):
            try:
                text = (stt.transcribe(path, language_code=lang).get("transcript") or "").strip()
                if text:
                    return text, lang
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [stt {lang} error] {e}")
                time.sleep(2)
    return "", "en-IN"


def speak(tts, text, out_wav, lang):
    for l in ([lang, "en-IN"] if lang != "en-IN" else ["en-IN"]):
        for _ in range(2):
            try:
                tts.synthesize(text, voice="Kaveri", model="timbre-v2.5",
                               language=l, output_file=out_wav)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"  [tts {l} error] {e}")
                time.sleep(2)
    return False


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    from src import qa
    stt, tts = GnaniSTTClient(), GnaniTTSClient()

    files = sorted(f for f in glob.glob(os.path.join(a.audio_dir, "*"))
                   if f.lower().endswith(EXTS))
    rows = []
    for n, path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"[{n}/{len(files)}] {base}")
        transcript, lang, reply, decision, sources = "", "en-IN", FALLBACK, "", []
        try:
            transcript, lang = transcribe(stt, path)
            if transcript:
                res = qa.rag_answer(transcript, k=5, use_rewrite=True)
                reply, decision, sources = res["reply"] or FALLBACK, res["decision"], res["sources"]
        except Exception as e:  # noqa: BLE001
            print(f"  [agent error] {e}")
        tts_lang = "hi-IN" if re.search(r"[\u0900-\u097f]", reply) else "en-IN"
        out_wav = os.path.join(a.out_dir, base + "_reply.wav")
        ok = speak(tts, clean_for_speech(reply), out_wav, tts_lang)
        with open(os.path.join(a.out_dir, base + "_reply.txt"), "w", encoding="utf-8") as f:
            f.write(reply)
        rows.append({"file": os.path.basename(path), "transcript": transcript,
                     "decision": decision, "reply_text": reply,
                     "reply_audio": os.path.basename(out_wav) if ok else "",
                     "chunks_used": " ; ".join(sources)})
        print(f"  heard: {transcript[:80]}\n  reply: {reply[:120]}")
    with open(os.path.join(a.out_dir, "results.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "transcript", "decision", "reply_text",
                                          "reply_audio", "chunks_used"])
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} files -> {a.out_dir}")


if __name__ == "__main__":
    main()