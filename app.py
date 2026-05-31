import os, io
from datetime import date
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from supabase import create_client
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis

load_dotenv()

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

DAILY_LIMITS = {
    "staff":       None,
    "hostel":      3,
    "day_scholar": 1
}

pending_image = None
last_result   = {}

face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    faces = face_app.get(img_array)
    if not faces:
        return None
    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_today_count(reg_number):
    today  = date.today().isoformat()
    result = supabase.table("entry_log") \
        .select("id") \
        .eq("reg_number", reg_number) \
        .eq("entry_date", today) \
        .execute()
    return len(result.data)

def log_entry(reg_number):
    today = date.today().isoformat()
    supabase.table("entry_log").insert({
        "reg_number": reg_number,
        "entry_date": today
    }).execute()

# ─── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    members = supabase.table("members").select("*").execute().data
    return render_template("index.html", members=members)

@app.route("/register_capture", methods=["POST"])
def register_capture():
    global pending_image
    pending_image = request.data
    return jsonify({"status": "AWAITING_DETAILS"}), 200

@app.route("/register_save", methods=["POST"])
def register_save():
    global pending_image
    if not pending_image:
        return jsonify({"error": "No pending image. Press register button first."}), 400

    data            = request.get_json()
    name            = data.get("name")
    reg_number      = data.get("reg_number")
    category        = data.get("category")
    last_date_str   = data.get("last_date")  # None for staff

    if not all([name, reg_number, category]):
        return jsonify({"error": "Missing fields"}), 400

    if category != "staff" and not last_date_str:
        return jsonify({"error": "Last date is required for this category"}), 400

    embedding = get_embedding(pending_image)
    if embedding is None:
        pending_image = None
        return jsonify({"error": "No face detected in image. Try again."}), 400

    img_path = os.path.join(KNOWN_FACES_DIR, f"{reg_number}.jpg")
    image = Image.open(io.BytesIO(pending_image)).convert("RGB")
    image.save(img_path)

    emb_path = os.path.join(KNOWN_FACES_DIR, f"{reg_number}.npy")
    np.save(emb_path, embedding)

    supabase.table("members").upsert({
        "reg_number":      reg_number,
        "name":            name,
        "category":        category,
        "registered_date": date.today().isoformat(),
        "last_date":       last_date_str if category != "staff" else None
    }).execute()

    pending_image = None
    return jsonify({"status": "REGISTERED", "name": name}), 200

@app.route("/renew", methods=["POST"])
def renew():
    data       = request.get_json()
    reg_number = data.get("reg_number")
    new_date   = data.get("new_date")

    if not all([reg_number, new_date]):
        return jsonify({"error": "Missing fields"}), 400

    supabase.table("members").update({
        "last_date": new_date
    }).eq("reg_number", reg_number).execute()

    return jsonify({"status": "RENEWED", "reg_number": reg_number, "new_date": new_date}), 200

@app.route("/remove_member", methods=["POST"])
def remove_member():
    data       = request.get_json()
    reg_number = data.get("reg_number")

    if not reg_number:
        return jsonify({"error": "Missing reg_number"}), 400

    # Delete entry logs first (foreign key)
    supabase.table("entry_log").delete().eq("reg_number", reg_number).execute()

    # Delete member
    supabase.table("members").delete().eq("reg_number", reg_number).execute()

    # Delete face files
    for ext in [".jpg", ".npy"]:
        path = os.path.join(KNOWN_FACES_DIR, f"{reg_number}{ext}")
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"status": "REMOVED"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    global last_result
    raw_bytes = request.data

    unknown_embedding = get_embedding(raw_bytes)
    if unknown_embedding is None:
        last_result = {"status": "UNKNOWN", "message": "No face detected"}
        return jsonify(last_result), 200

    members = supabase.table("members").select("*").execute().data
    if not members:
        last_result = {"status": "UNKNOWN", "message": "No registered members"}
        return jsonify(last_result), 200

    best_match = None
    best_score = -1

    for member in members:
        emb_path = os.path.join(KNOWN_FACES_DIR, f"{member['reg_number']}.npy")
        if not os.path.exists(emb_path):
            continue
        known_embedding = np.load(emb_path)
        score = cosine_similarity(unknown_embedding, known_embedding)
        if score > best_score:
            best_score = score
            best_match = member

    print(f"Best score: {best_score} for {best_match}")

    if best_match is None or best_score < 0.3:
        last_result = {"status": "UNKNOWN", "message": "Face not recognised"}
        return jsonify(last_result), 200

    reg_number = best_match["reg_number"]
    name       = best_match["name"]
    category   = best_match["category"]
    last_date  = best_match.get("last_date")
    limit      = DAILY_LIMITS[category]

    # Check expiry for non-staff
    if category != "staff" and last_date:
        if date.today().isoformat() > last_date:
            last_result = {
                "status":     "EXPIRED",
                "name":       name,
                "reg":        reg_number,
                "message":    "Membership expired! Renewal required.",
                "last_date":  last_date
            }
            return jsonify(last_result), 200

    # Check daily limit
    if limit is not None:
        count = get_today_count(reg_number)
        if count >= limit:
            last_result = {
                "status":  "DENIED",
                "name":    name,
                "reg":     reg_number,
                "message": f"Daily limit of {limit} reached"
            }
            return jsonify(last_result), 200

    log_entry(reg_number)
    last_result = {
        "status":   "MATCH",
        "name":     name,
        "reg":      reg_number,
        "category": category
    }
    return jsonify(last_result), 200

@app.route("/members", methods=["GET"])
def get_members():
    members = supabase.table("members").select("*").execute().data
    return jsonify(members)

@app.route("/last_recognition")
def last_recognition():
    return jsonify(last_result)

@app.route("/pending_status")
def pending_status():
    return jsonify({"pending": pending_image is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)