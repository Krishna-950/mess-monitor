import os, io, json
from datetime import date
from flask import Flask, request, jsonify, render_template
import face_recognition
import numpy as np
from PIL import Image
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─── Supabase Setup ─────────────────────────────────────────────
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

# ─── Helpers ────────────────────────────────────────────────────
def load_all_known_faces():
    encodings, ids = [], []
    members = supabase.table("members").select("*").execute().data
    for m in members:
        img_path = os.path.join(KNOWN_FACES_DIR, f"{m['reg_number']}.jpg")
        if os.path.exists(img_path):
            image = face_recognition.load_image_file(img_path)
            encs  = face_recognition.face_encodings(image)
            if encs:
                encodings.append(encs[0])
                ids.append(m['reg_number'])
    return encodings, ids

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

def decode_image(raw_bytes):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.array(image)

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

    data       = request.get_json()
    name       = data.get("name")
    reg_number = data.get("reg_number")
    category   = data.get("category")

    if not all([name, reg_number, category]):
        return jsonify({"error": "Missing fields"}), 400

    img_path = os.path.join(KNOWN_FACES_DIR, f"{reg_number}.jpg")
    with open(img_path, "wb") as f:
        f.write(pending_image)

    image = face_recognition.load_image_file(img_path)
    if not face_recognition.face_encodings(image):
        os.remove(img_path)
        pending_image = None
        return jsonify({"error": "No face detected in image. Try again."}), 400

    supabase.table("members").upsert({
        "reg_number": reg_number,
        "name":       name,
        "category":   category
    }).execute()

    pending_image = None
    return jsonify({"status": "REGISTERED", "name": name}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    global last_result
    raw_bytes     = request.data
    unknown_image = decode_image(raw_bytes)
    unknown_encs  = face_recognition.face_encodings(unknown_image)

    if not unknown_encs:
        last_result = {"status": "UNKNOWN", "message": "No face detected"}
        return jsonify(last_result), 200

    known_encodings, known_ids = load_all_known_faces()
    if not known_encodings:
        last_result = {"status": "UNKNOWN", "message": "No registered members"}
        return jsonify(last_result), 200

    distances = face_recognition.face_distance(known_encodings, unknown_encs[0])
    best_idx  = int(np.argmin(distances))

    if distances[best_idx] > 0.5:
        last_result = {"status": "UNKNOWN", "message": "Face not recognised"}
        return jsonify(last_result), 200

    reg_number = known_ids[best_idx]
    member     = supabase.table("members").select("*").eq("reg_number", reg_number).execute().data[0]
    name       = member["name"]
    category   = member["category"]
    limit      = DAILY_LIMITS[category]

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