from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

import os
import json
import threading
import time

from pymongo import MongoClient

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- APP ----------------
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
MONGODB_URI = os.getenv("MONGODB_URI") or "mongodb+srv://admin:<db_password>@cluster0.et1jtcv.mongodb.net/?appName=Cluster0"

USERS_FILE = os.path.join(os.path.dirname(__file__), "users_store.json")
ALERTS_FILE = os.path.join(os.path.dirname(__file__), "alerts_store.json")
_store_lock = threading.RLock()

# Throttle alert persistence to avoid flooding (seconds between same-type alerts)
_alert_last_stored = {}  # type -> timestamp

mongo_available = False
alerts_collection = None
mongo_error = None
client = None
users_collection = None

try:
    # Avoid accidentally using the placeholder creds in the repo.
    if "<db_password>" in MONGODB_URI:
        raise ValueError("MONGODB_URI contains placeholder <db_password>")

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    # Force an auth/connection check now, so we can fallback cleanly.
    client.admin.command("ping")

    db = client["ai_dashboard"]
    users_collection = db["users"]
    alerts_collection = db["alerts"]
    mongo_available = True
except Exception as e:
    mongo_error = str(e)
    mongo_available = False
    alerts_collection = None


def _load_users_file():
    with _store_lock:
        if not os.path.exists(USERS_FILE):
            return {"users": []}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("users"), list):
            return data
        return {"users": []}


def _save_users_file(data):
    with _store_lock:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)


def _file_find_one(username: str):
    data = _load_users_file()
    for u in data.get("users", []):
        if u.get("username") == username:
            return u
    return None


def _file_insert_one(user_doc: dict):
    data = _load_users_file()
    for u in data.get("users", []):
        if u.get("username") == user_doc.get("username"):
            raise ValueError("User exists")
    data["users"].append(user_doc)
    _save_users_file(data)


def _file_delete_many():
    data = _load_users_file()
    deleted = len(data.get("users", []))
    data["users"] = []
    _save_users_file(data)
    return deleted


def _file_delete_user(username: str):
    data = _load_users_file()
    original_len = len(data.get("users", []))
    data["users"] = [u for u in data.get("users", []) if u.get("username") != username]
    _save_users_file(data)
    return 1 if len(data["users"]) < original_len else 0


def _file_list_users():
    data = _load_users_file()
    return [{"username": u.get("username"), "role": u.get("role", "user")} for u in data.get("users", [])]


def _load_alerts_file():
    with _store_lock:
        if not os.path.exists(ALERTS_FILE):
            return {"alerts": []}
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("alerts"), list):
                return data
        except (json.JSONDecodeError, IOError):
            pass
        return {"alerts": []}


def _save_alerts_file(data):
    with _store_lock:
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)


def _file_insert_alert(alert_doc: dict):
    data = _load_alerts_file()
    doc = {k: v for k, v in alert_doc.items() if k != "_id"}
    data["alerts"].append(doc)
    _save_alerts_file(data)


def _file_get_alerts():
    data = _load_alerts_file()
    return data.get("alerts", [])


def find_user(username: str):
    if mongo_available:
        return users_collection.find_one({"username": username})
    return _file_find_one(username)


def insert_user(user_doc: dict):
    if mongo_available:
        users_collection.insert_one(user_doc)
        return
    _file_insert_one(user_doc)


def delete_all_users():
    if mongo_available:
        return users_collection.delete_many({}).deleted_count
    return _file_delete_many()


def delete_user_by_username(username: str):
    if mongo_available:
        r = users_collection.delete_one({"username": username})
        return r.deleted_count
    return _file_delete_user(username)


def list_all_users():
    if mongo_available:
        return list(users_collection.find({}, {"username": 1, "role": 1, "_id": 0}))
    return _file_list_users()


def insert_alert(alert_doc: dict):
    if mongo_available and alerts_collection is not None:
        alerts_collection.insert_one(alert_doc)
    else:
        _file_insert_alert(alert_doc)


def get_all_alerts():
    if mongo_available and alerts_collection is not None:
        return list(alerts_collection.find({}, {"_id": 0}))
    return _file_get_alerts()

# ---------------- AUTH ----------------
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# bcrypt dependency/backend appears broken in this environment; use pbkdf2 for reliability.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    if not hashed:
        return False
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------- AUTH ROUTES ----------------


@app.post("/register")
def register(username: str, password: str, role: str = "user"):
    try:
        if find_user(username):
            raise HTTPException(status_code=400, detail="User exists")

        insert_user({
            "username": username,
            "password": hash_password(password),
            "role": role
        })

        return {"msg": "User created"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_users():
    """
    Clears all users.
    Use this when stored hashes are incompatible/corrupted or the DB/file store has bad data.
    """
    try:
        deleted = delete_all_users()
        storage = "mongo" if mongo_available else "file"
        return {"msg": "Users cleared", "deleted": deleted, "storage": storage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login")
def login(username: str, password: str):
    try:
        user = find_user(username)

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        hashed_password = user.get("password")
        if not verify_password(password, hashed_password):
            raise HTTPException(status_code=401, detail="Wrong password")

        token = create_access_token({
            "sub": username,
            "role": user.get("role", "user")
        })

        return {"access_token": token}

    except Exception as e:
        print("LOGIN ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin")
def admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not allowed")

    return {"msg": "Welcome Admin"}


@app.get("/users")
def get_users(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not allowed")
    return list_all_users()


@app.delete("/delete-user/{username}")
def delete_user(username: str, user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not allowed")
    deleted = delete_user_by_username(username)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"msg": "User deleted"}


@app.get("/alerts")
def get_alerts(user=Depends(get_current_user)):
    alerts = get_all_alerts()
    return list(reversed(alerts[-100:]))  # last 100, newest first


# ---------------- AI SYSTEM ----------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

people_count_global = 0
alerts_global = []


def generate_frames():
    global people_count_global, alerts_global

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            if int(class_id) == 0:
                detections.append(([x1, y1, x2-x1, y2-y1], conf, class_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        people_count_global = len(tracks)
        alerts_global = []
        now_ts = datetime.now()
        throttle_sec = 5
        now_time = time.time()

        if people_count_global > 1:
            alerts_global.append("🚨 CROWD ALERT")
            last = _alert_last_stored.get("CROWD", 0)
            if now_time - last >= throttle_sec:
                insert_alert({"type": "CROWD", "time": now_ts.isoformat(), "count": people_count_global})
                _alert_last_stored["CROWD"] = now_time

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, w, h = track.to_ltrb()
            cx = int(l + w / 2)
            cy = int(t + h / 2)

            if 100 < cx < 400 and 100 < cy < 400:
                alerts_global.append("🚨 INTRUSION ALERT")
                last = _alert_last_stored.get("INTRUSION", 0)
                if now_time - last >= throttle_sec:
                    insert_alert({"type": "INTRUSION", "time": now_ts.isoformat(), "track_id": str(track.track_id)})
                    _alert_last_stored["INTRUSION"] = now_time

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video")
def video():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats():
    return {"people": people_count_global, "alerts": alerts_global}


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_json({
            "people": people_count_global,
            "alerts": alerts_global
        })
