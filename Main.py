from flask import Flask, request, jsonify, send_from_directory
import os, uuid, tempfile, threading, time
from huggingface_hub import InferenceClient

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise Exception("Please set HF_TOKEN in Repl secrets")

# Clients: text LLM and provider Novita for video
hf_text = InferenceClient(token=HF_TOKEN)
hf_video = InferenceClient(provider="novita", token=HF_TOKEN)

GPT_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"

# Simple in-memory job store: job_id -> status dict
jobs = {}  # { job_id: {"status": "queued|processing|done|error", "message": "", "download_url": "", "script": ""} }

def safe_extract_text(resp):
    # InferenceClient may return str or dict/list; try to extract text
    try:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict) and "generated_text" in resp:
            return resp["generated_text"]
        if isinstance(resp, list) and len(resp) > 0:
            if isinstance(resp[0], dict):
                return resp[0].get("generated_text", str(resp[0]))
            return str(resp[0])
        return str(resp)
    except Exception:
        return str(resp)

def generation_worker(job_id, topic, video_length, ratio, base_url):
    jobs[job_id]["status"] = "processing"
    try:
        # 1) LLM: generate script
        prompt = (
            f"Generate a concise, engaging, viral YouTube Shorts script (max {video_length} seconds) on the topic: '{topic}'.\n"
            "Make it faceless (no human faces). Include a strong hook, 2-3 clear points, short visual cues for each line (e.g., 'close-up of object', 'text overlay', 'stock nature b-roll'), and a call to action. Format for vertical 9:16 short."
        )
        text_resp = hf_text.text_generation(model=GPT_MODEL, inputs=prompt, max_new_tokens=400, temperature=0.7)
        script = safe_extract_text(text_resp)
        jobs[job_id]["script"] = script

        # 2) Video: ask video model to create faceless vertical video using the script as instruction
        video_prompt = (
            f"{script}\n\nVideo instructions: vertical {ratio}, no faces, fast cuts, text overlays, length ~{video_length}s. Keep visuals high-energy and suitable for a YouTube Short."
        )

        video_bytes = hf_video.text_to_video(video_prompt, model=VIDEO_MODEL)

        # 3) Save file
        filename = f"short_{uuid.uuid4().hex}.mp4"
        tmp_dir = tempfile.gettempdir()
        filepath = os.path.join(tmp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(video_bytes)

        # 4) Compose download URL
        download_url = base_url.rstrip("/") + "/download/" + filename
        jobs[job_id]["status"] = "done"
        jobs[job_id]["download_url"] = download_url
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)

@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    topic = body.get("topic", "").strip()
    video_length = int(body.get("video_length", 60))
    ratio = body.get("ratio", "9:16")

    if not topic:
        return jsonify({"status": "error", "message": "Missing topic"}), 400

    job_id = uuid.uuid4().hex
    # store job and return immediately
    jobs[job_id] = {"status": "queued", "message": "", "download_url": "", "script": ""}

    base_url = request.url_root.rstrip("/")

    # start background worker
    t = threading.Thread(target=generation_worker, args=(job_id, topic, video_length, ratio, base_url))
    t.daemon = True
    t.start()

    return jsonify({
        "status": "submitted",
        "job_id": job_id,
        "status_url": base_url + "/status/" + job_id
    }), 202

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    j = jobs.get(job_id)
    if not j:
        return jsonify({"status": "error", "message": "job_id not found"}), 404
    return jsonify(j)

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    tmp_dir = tempfile.gettempdir()
    return send_from_directory(tmp_dir, filename, as_attachment=True, mimetype="video/mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
