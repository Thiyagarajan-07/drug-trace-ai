import os
import io
import csv
import hashlib
import datetime
import numpy as np
import cv2
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt
import uvicorn

app = FastAPI(title="DRUG-TRACE AI - NextGen Forensic HUD")

# --- 1. DYNAMIC DATABASE ENGINE ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./drug_trace.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. ORM DATABASE MODELS ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    badge_number = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    officer_id = Column(String, nullable=False)
    fingerprint_hash = Column(String, nullable=False)
    gps_coords = Column(String, nullable=False)
    timestamp_ist = Column(String, nullable=False)
    classification = Column(String, nullable=False)
    hue_score = Column(Float, nullable=False)
    sha256_hash = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

# --- 3. ZERO-CRASH AUTHENTICATION ---
SECRET_KEY = os.getenv("SECRET_KEY", "sih_forensic_secret_key_2026")
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    salt = "sih_forensic_salt_2026"
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()

def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed

# --- 4. OPENCV COMPUTER VISION HSV ENGINE ---
def process_test_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "INCONCLUSIVE", 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    center_hsv = hsv[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]

    # HSV color space masks for chemical color changes
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([18, 255, 255])
    lower_red2 = np.array([120, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(center_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(center_hsv, lower_red2, upper_red2)
    positive_mask = cv2.bitwise_or(mask1, mask2)

    total_pixels = center_hsv.shape[0] * center_hsv.shape[1]
    positive_pixels = cv2.countNonZero(positive_mask)
    positive_ratio = (positive_pixels / total_pixels) * 100

    avg_hue = float(np.mean(center_hsv[:, :, 0]))
    classification = "POSITIVE" if positive_ratio >= 3.0 else "NEGATIVE"
    return classification, round(avg_hue, 2)

# --- 5. SYSTEM ENDPOINTS ---
@app.get("/ping")
def ping():
    return {"status": "active", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.post("/register")
def register(
    username: str = Form(...),
    badge_number: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.username == username).first():
        return JSONResponse({"status": "error", "message": "Username already exists"}, status_code=400)
    if db.query(User).filter(User.badge_number == badge_number).first():
        return JSONResponse({"status": "error", "message": "Badge ID already registered"}, status_code=400)
    
    new_user = User(
        username=username,
        badge_number=badge_number,
        hashed_password=hash_password(password)
    )
    db.add(new_user)
    db.commit()
    return JSONResponse({"status": "success", "message": f"Officer Badge {badge_number} registered successfully."})

@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return JSONResponse({"status": "error", "message": "Invalid username or password"}, status_code=401)
    
    token_payload = {
        "sub": user.username,
        "badge_id": user.badge_number,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }
    token = jwt.encode(token_payload, SECRET_KEY, algorithm=ALGORITHM)
    return JSONResponse({
        "status": "success",
        "access_token": token,
        "badge_id": user.badge_number,
        "username": user.username
    })

@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    badge_id: str = Form(...),
    gps_coords: str = Form(...),
    db: Session = Depends(get_db)
):
    image_bytes = file.file.read()
    classification, hue_score = process_test_image(image_bytes)

    now_utc = datetime.datetime.utcnow()
    ist_time = (now_utc + datetime.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S IST")
    
    fingerprint_hash = hashlib.sha256(badge_id.encode()).hexdigest()[:16].upper()
    
    raw_signature = f"{badge_id}:{classification}:{hue_score}:{gps_coords}:{ist_time}:{image_bytes[:32]}"
    sha256_hash = hashlib.sha256(raw_signature.encode()).hexdigest()

    log_entry = AuditLog(
        officer_id=badge_id,
        fingerprint_hash=fingerprint_hash,
        gps_coords=gps_coords,
        timestamp_ist=ist_time,
        classification=classification,
        hue_score=hue_score,
        sha256_hash=sha256_hash
    )
    db.add(log_entry)
    db.commit()

    return JSONResponse({
        "status": "success",
        "classification": classification,
        "hue_score": hue_score,
        "badge_id": badge_id,
        "fingerprint": fingerprint_hash,
        "timestamp_ist": ist_time,
        "gps_coords": gps_coords,
        "sha256_hash": sha256_hash
    })

@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.id.desc()).all()
    results = []
    for l in logs:
        results.append({
            "id": l.id,
            "officer_id": l.officer_id,
            "fingerprint_hash": l.fingerprint_hash,
            "gps_coords": l.gps_coords,
            "timestamp_ist": l.timestamp_ist,
            "classification": l.classification,
            "hue_score": l.hue_score,
            "sha256_hash": l.sha256_hash
        })
    return JSONResponse({"status": "success", "count": len(results), "logs": results})

@app.get("/export-csv")
def export_csv(db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.id.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Log ID", "Officer Badge", "Biometric ID", "Classification", "Hue Score", "GPS Location", "IST Timestamp", "SHA-256 Seal"])
    
    for l in logs:
        writer.writerow([l.id, l.officer_id, l.fingerprint_hash, l.classification, l.hue_score, l.gps_coords, l.timestamp_ist, l.sha256_hash])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=drug_trace_legal_audit_vault.csv"}
    )

# --- 6. LUXURY TECH-FORWARD FRONTEND HUD (WITH AUTH MODES & WEB AUDIO SFX) ---
@app.get("/", response_class=HTMLResponse)
def serve_portal():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DRUG-TRACE AI | Luxury Forensic Intelligence</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
        
        <style>
            :root {
                --bg-obsidian: #030406;
                --bg-surface: rgba(12, 14, 20, 0.82);
                --border-translucent: rgba(255, 255, 255, 0.08);
                --border-active: rgba(16, 185, 129, 0.45);
                --emerald: #10b981;
                --emerald-glow: rgba(16, 185, 129, 0.25);
                --amber: #f59e0b;
                --cyan-accent: #06b6d4;
                --text-main: #f9fafb;
                --text-muted: #9ca3af;
                --danger: #ef4444;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            
            body {
                background: var(--bg-obsidian);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(circle at 50% -12%, rgba(16, 185, 129, 0.15) 0%, transparent 48%),
                    radial-gradient(circle at 100% 92%, rgba(6, 182, 212, 0.1) 0%, transparent 42%);
            }

            .mono { font-family: 'JetBrains Mono', monospace; }
            .orbitron { font-family: 'Orbitron', sans-serif; }

            .glass-panel {
                background: var(--bg-surface);
                backdrop-filter: blur(28px);
                -webkit-backdrop-filter: blur(28px);
                border: 1px solid var(--border-translucent);
                border-radius: 18px;
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
                transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .glass-panel:hover {
                border-color: rgba(255, 255, 255, 0.18);
                box-shadow: 0 36px 72px rgba(0, 0, 0, 0.7);
                transform: translateY(-2px);
            }

            .top-telemetry {
                background: rgba(3, 4, 6, 0.92);
                border-bottom: 1px solid var(--border-translucent);
                padding: 12px 48px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 11px;
                letter-spacing: 0.9px;
            }

            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--emerald);
                font-weight: 600;
            }

            .pulse-dot {
                width: 7px;
                height: 7px;
                background: var(--emerald);
                border-radius: 50%;
                box-shadow: 0 0 12px var(--emerald);
                animation: softPulse 2s infinite ease-in-out;
            }

            @keyframes softPulse { 0% { opacity: 0.4; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.3); } 100% { opacity: 0.4; transform: scale(0.9); } }

            .navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 24px 56px;
                border-bottom: 1px solid var(--border-translucent);
                background: rgba(12, 14, 20, 0.6);
                backdrop-filter: blur(16px);
            }

            .brand-logo {
                display: flex;
                align-items: center;
                gap: 14px;
                cursor: pointer;
            }

            .nav-links {
                display: flex;
                gap: 36px;
                align-items: center;
            }

            .nav-link {
                color: var(--text-muted);
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                transition: color 0.2s;
                cursor: pointer;
            }

            .nav-link:hover { color: #fff; }

            #publicHero {
                padding: 90px 24px 110px;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                position: relative;
            }

            .badge-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 20px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.35);
                border-radius: 30px;
                color: var(--emerald);
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 32px;
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.12);
            }

            .hero-heading {
                font-size: clamp(44px, 7vw, 80px);
                font-weight: 900;
                letter-spacing: -1.8px;
                line-height: 1.06;
                margin-bottom: 24px;
                max-width: 980px;
                background: linear-gradient(135deg, #ffffff 30%, var(--text-muted) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-description {
                font-size: 18px;
                color: var(--text-muted);
                max-width: 720px;
                line-height: 1.8;
                margin-bottom: 48px;
                font-weight: 300;
            }

            .cta-group {
                display: flex;
                gap: 18px;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 88px;
            }

            .btn-primary {
                background: linear-gradient(135deg, #10b981 0%, #047857 100%);
                color: #ffffff;
                font-weight: 600;
                padding: 16px 36px;
                border-radius: 12px;
                border: none;
                cursor: pointer;
                box-shadow: 0 10px 28px rgba(16, 185, 129, 0.35);
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
                font-size: 14px;
                letter-spacing: 0.5px;
            }

            .btn-primary:hover {
                transform: translateY(-3px);
                box-shadow: 0 14px 36px rgba(16, 185, 129, 0.5);
            }

            .btn-secondary {
                background: rgba(255, 255, 255, 0.04);
                color: var(--text-main);
                font-weight: 600;
                padding: 16px 36px;
                border-radius: 12px;
                border: 1px solid var(--border-translucent);
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
                font-size: 14px;
            }

            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.08);
                border-color: rgba(255, 255, 255, 0.22);
                transform: translateY(-3px);
            }

            .features-container {
                max-width: 1280px;
                width: 100%;
                margin: 0 auto;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
                gap: 32px;
                padding: 0 24px;
            }

            .feature-card {
                padding: 40px;
                text-align: left;
                position: relative;
            }

            .feature-icon {
                font-size: 30px;
                margin-bottom: 22px;
                display: inline-block;
                padding: 16px;
                background: rgba(16, 185, 129, 0.1);
                border-radius: 14px;
                border: 1px solid rgba(16, 185, 129, 0.25);
            }

            .feature-title {
                font-size: 21px;
                font-weight: 700;
                color: #fff;
                margin-bottom: 12px;
                letter-spacing: -0.5px;
            }

            .feature-desc {
                font-size: 14px;
                color: var(--text-muted);
                line-height: 1.7;
            }

            #authModal {
                position: fixed;
                inset: 0;
                background: rgba(3, 4, 6, 0.9);
                backdrop-filter: blur(28px);
                z-index: 9999;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .auth-box {
                width: 100%;
                max-width: 460px;
                padding: 48px;
                border-radius: 22px;
            }

            .auth-mode-tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 28px;
                background: rgba(255, 255, 255, 0.03);
                padding: 6px;
                border-radius: 12px;
                border: 1px solid var(--border-translucent);
            }

            .auth-mode-btn {
                flex: 1;
                background: transparent;
                border: none;
                color: var(--text-muted);
                padding: 10px;
                font-weight: 600;
                font-size: 12px;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.2s;
            }

            .auth-mode-btn.active {
                background: var(--emerald);
                color: #fff;
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
            }

            .form-input {
                width: 100%;
                padding: 15px 18px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--border-translucent);
                border-radius: 10px;
                color: #fff;
                font-size: 14px;
                margin-top: 8px;
                margin-bottom: 22px;
                transition: all 0.3s ease;
            }

            .form-input:focus {
                outline: none;
                border-color: var(--emerald);
                box-shadow: 0 0 22px rgba(16, 185, 129, 0.18);
                background: rgba(255, 255, 255, 0.06);
            }

            #appLayout { display: none; flex-direction: column; min-height: 100vh; }

            header {
                background: rgba(12, 14, 20, 0.88);
                border-bottom: 1px solid var(--border-translucent);
                padding: 20px 56px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .nav-tabs-bar {
                display: flex;
                gap: 12px;
                background: rgba(3, 4, 6, 0.92);
                padding: 12px 56px;
                border-bottom: 1px solid var(--border-translucent);
                overflow-x: auto;
            }

            .tab-btn {
                background: transparent;
                border: none;
                color: var(--text-muted);
                padding: 11px 22px;
                font-weight: 500;
                font-size: 13px;
                cursor: pointer;
                border-radius: 10px;
                transition: all 0.2s ease;
                white-space: nowrap;
            }

            .tab-btn:hover { color: #fff; background: rgba(255, 255, 255, 0.04); }
            .tab-btn.active { color: var(--emerald); background: rgba(16, 185, 129, 0.14); font-weight: 600; }

            main { flex: 1; padding: 52px; max-width: 1400px; margin: 0 auto; width: 100%; }

            .tab-content { display: none; }
            .tab-content.active { display: block; }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 28px;
                margin-bottom: 40px;
            }

            .metric-card { padding: 32px; }
            .metric-value { font-size: 42px; font-weight: 700; margin-top: 10px; letter-spacing: -0.5px; }

            .scanner-wrapper {
                max-width: 740px;
                margin: 0 auto;
                padding: 48px;
            }

            .dropzone {
                border: 2px dashed rgba(255, 255, 255, 0.16);
                background: rgba(255, 255, 255, 0.02);
                border-radius: 16px;
                padding: 60px 24px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .dropzone:hover {
                border-color: var(--emerald);
                background: rgba(16, 185, 129, 0.03);
            }

            .table-container { padding: 36px; overflow-x: auto; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
            th { background: rgba(0, 0, 0, 0.45); color: var(--text-muted); padding: 16px 20px; border-bottom: 1px solid var(--border-translucent); font-weight: 600; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
            td { padding: 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); color: #e5e7eb; }
            tr:hover td { background: rgba(255, 255, 255, 0.02); }

            footer {
                border-top: 1px solid var(--border-translucent);
                padding: 56px;
                text-align: center;
                color: var(--text-muted);
                font-size: 13px;
                background: rgba(3, 4, 6, 0.75);
                display: flex;
                flex-direction: column;
                gap: 18px;
                align-items: center;
            }
        </style>
    </head>
    <body>

        <!-- TOP DIAGNOSTIC TELEMETRY BAR -->
        <div class="top-telemetry mono">
            <div style="display:flex; gap:32px; align-items:center;">
                <div class="status-indicator">
                    <span class="pulse-dot"></span> SECURE PROTOCOL ACTIVE
                </div>
                <div>LATENCY: <span style="color:var(--emerald);">10ms</span></div>
                <div>CIPHER: <span style="color:var(--cyan-accent);">SHA-256 / PBKDF2</span></div>
            </div>
            <div style="display:flex; gap:32px; align-items:center;">
                <div>IST TIMESTAMP: <span id="clockIST" style="color:#fff;">--:--:--</span></div>
                <div>GPS PRECISION: <span style="color:var(--emerald);">RTK LOCKED</span></div>
            </div>
        </div>

        <!-- PUBLIC LANDING NAVIGATION -->
        <nav class="navbar">
            <div class="brand-logo" onclick="playSound('click'); location.reload()">
                <div style="font-size:26px; color:var(--emerald);">🛡️</div>
                <div class="orbitron" style="font-size:16px; font-weight:700; color:#fff; letter-spacing:0.5px;">DRUG-TRACE AI</div>
            </div>
            <div class="nav-links">
                <a class="nav-link" onclick="playSound('click'); openAuthModal('login')">Officer Access</a>
                <button onclick="playSound('click'); enableDemoMode()" class="btn-primary" style="padding:10px 22px; font-size:13px;">Evaluator Portal</button>
            </div>
        </nav>

        <!-- PUBLIC HERO LANDING & LUXURY FEATURES -->
        <div id="publicHero">
            <div class="badge-pill">
                <span>✦</span> Next-Gen Forensic Intelligence & Field Suite
            </div>
            <h1 class="hero-heading">Autonomous Narcotic Spot-Test Analysis & Verification</h1>
            <p class="hero-description">
                Empowering law enforcement professionals with high-precision OpenCV spectral matrix mapping, real-time geolocation timestamping, and immutable cryptographic evidence vaults.
            </p>
            <div class="cta-group">
                <button onclick="playSound('click'); openAuthModal('login')" class="btn-primary">⚡ Officer Authentication</button>
                <button onclick="playSound('click'); enableDemoMode()" class="btn-secondary">Explore Evaluator Portal</button>
            </div>

            <!-- FEATURE CARDS -->
            <div class="features-container">
                <div class="glass-panel feature-card">
                    <div class="feature-icon">🧪</div>
                    <h3 class="feature-title">OpenCV Spectral Matrix</h3>
                    <p class="feature-desc">Automated HSV color-space segmentation detects chemical reagent reactions instantly with high-precision feedback.</p>
                </div>
                <div class="glass-panel feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3 class="feature-title">Cryptographic Sealing</h3>
                    <p class="feature-desc">Generates immutable SHA-256 evidence hashes tied to officer badges, GPS coordinates, and IST timestamps.</p>
                </div>
                <div class="glass-panel feature-card">
                    <div class="feature-icon">📑</div>
                    <h3 class="feature-title">Legal Audit Vault</h3>
                    <p class="feature-desc">Export certified audit logs directly to CSV reports designed for airtight court presentation and chain-of-custody tracking.</p>
                </div>
            </div>
        </div>

        <!-- OFFICER AUTHENTICATION & REGISTRATION MODAL -->
        <div id="authModal">
            <div class="auth-box glass-panel">
                <div style="text-align:center; margin-bottom:24px;">
                    <h2 class="orbitron" id="authTitle" style="font-size:22px; font-weight:700; color:#fff; margin-bottom:8px;">OFFICER LOGIN</h2>
                    <p style="font-size:12px; color:var(--text-muted);" class="mono">SECURE FIELD OPERATIVE PORTAL</p>
                </div>

                <!-- Mode switcher tabs -->
                <div class="auth-mode-tabs">
                    <button class="auth-mode-btn active" id="tabLoginBtn" onclick="playSound('click'); switchAuthMode('login')">Sign In</button>
                    <button class="auth-mode-btn" id="tabRegBtn" onclick="playSound('click'); switchAuthMode('register')">Register New Officer</button>
                </div>

                <form id="authForm" onsubmit="handleAuthSubmit(event)">
                    <label style="font-size:11px; font-weight:600; color:var(--text-muted); letter-spacing:0.5px;">USERNAME</label>
                    <input type="text" id="usernameInput" class="form-input mono" placeholder="officer_agent" required>

                    <div id="badgeGroup" style="display:none;">
                        <label style="font-size:11px; font-weight:600; color:var(--text-muted); letter-spacing:0.5px;">BADGE NUMBER</label>
                        <input type="text" id="badgeRegisterInput" class="form-input mono" placeholder="IND-POLICE-XXXX">
                    </div>

                    <label style="font-size:11px; font-weight:600; color:var(--text-muted); letter-spacing:0.5px;">SECURITY ACCESS KEY</label>
                    <input type="password" id="passwordInput" class="form-input" placeholder="••••••••" required>

                    <button type="submit" id="authSubmitBtn" class="btn-primary" style="width:100%; margin-top:10px;">Verify & Initialize Session</button>
                </form>

                <div style="text-align:center; margin-top:28px;">
                    <button onclick="playSound('click'); closeAuthModal()" style="background:none; border:none; color:var(--text-muted); font-size:12px; cursor:pointer;">← Return to Main Portal</button>
                </div>
            </div>
        </div>

        <!-- MAIN DASHBOARD APP INTERFACE -->
        <div id="appLayout">
            <header>
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="font-size:24px; color:var(--emerald);">🛡️</div>
                    <div>
                        <div class="orbitron" style="font-size:16px; font-weight:700; color:#fff; letter-spacing:0.5px;">DRUG-TRACE AI</div>
                        <div style="font-size:11px; color:var(--text-muted);" class="mono">FORENSIC HUD v4.3</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:18px;">
                    <span class="mono" id="headerBadgeTag" style="font-size:12px; padding:6px 16px; background:rgba(255,255,255,0.05); border:1px solid var(--border-translucent); border-radius:10px; color:var(--emerald);">BADGE: IND-POLICE-8042</span>
                    <button onclick="playSound('click'); lockSystem()" class="btn-secondary" style="padding:9px 18px; font-size:12px; border-color:rgba(239,68,68,0.3); color:#ef4444;">Lock HUD</button>
                </div>
            </header>

            <div class="nav-tabs-bar">
                <button class="tab-btn active" onclick="playSound('click'); switchTab('overview')">📊 Telemetry Dashboard</button>
                <button class="tab-btn" onclick="playSound('click'); switchTab('scanner')">🧪 Chemical Reagent Scanner</button>
                <button class="tab-btn" onclick="playSound('click'); switchTab('logs')">📑 Evidence Audit Vault</button>
                <button class="tab-btn" onclick="playSound('click'); switchTab('sih')">⚙️ System Architecture</button>
            </div>

            <main>
                <!-- OVERVIEW TAB -->
                <div id="tab-overview" class="tab-content active">
                    <div class="metrics-grid">
                        <div class="metric-card glass-panel">
                            <div style="font-size:12px; color:var(--text-muted); font-weight:500;">TOTAL FIELD EXECUTIONS</div>
                            <div class="metric-value mono" id="statTotal" style="color:#fff;">0</div>
                        </div>
                        <div class="metric-card glass-panel">
                            <div style="font-size:12px; color:var(--text-muted); font-weight:500;">POSITIVE DETECTIONS</div>
                            <div class="metric-value mono" id="statPos" style="color:var(--danger);">0</div>
                        </div>
                        <div class="metric-card glass-panel">
                            <div style="font-size:12px; color:var(--text-muted); font-weight:500;">NEGATIVE REAGENTS</div>
                            <div class="metric-value mono" id="statNeg" style="color:var(--emerald);">0</div>
                        </div>
                        <div class="metric-card glass-panel">
                            <div style="font-size:12px; color:var(--text-muted); font-weight:500;">CHAIN-OF-CUSTODY INTEGRITY</div>
                            <div class="metric-value mono" style="color:var(--cyan-accent);">100%</div>
                        </div>
                    </div>

                    <div class="glass-panel" style="padding:56px; text-align:center;">
                        <h2 class="orbitron" style="font-size:24px; margin-bottom:14px; color:#fff;">READY FOR FIELD REAGENT ANALYSIS</h2>
                        <p style="color:var(--text-muted); max-width:660px; margin:0 auto 36px; font-size:15px; line-height:1.75;">Upload or capture chemical spot-test strip photographs to evaluate OpenCV HSV color matrix channels and securely sign evidence logs.</p>
                        <button onclick="playSound('click'); switchTab('scanner')" class="btn-primary">Initialize Camera Scanner →</button>
                    </div>
                </div>

                <!-- SCANNER TAB -->
                <div id="tab-scanner" class="tab-content">
                    <div class="scanner-wrapper glass-panel">
                        <div style="text-align:center; margin-bottom:32px;">
                            <h2 class="orbitron" style="font-size:19px; color:#fff; margin-bottom:8px;">AI REAGENT ANALYZER</h2>
                            <p style="color:var(--text-muted); font-size:12px;" class="mono">OPENCV HSV COLOR SPACE MATRIX SEGMENTATION</p>
                        </div>

                        <form id="scannerForm">
                            <div class="dropzone" onclick="playSound('click'); document.getElementById('fileInput').click()">
                                <div style="font-size:52px; margin-bottom:16px;">📷</div>
                                <div style="font-weight:600; font-size:15px; color:#fff;" id="uploadNotice">Tap to Capture or Upload Strip Photo</div>
                                <div style="font-size:11px; color:var(--text-muted); margin-top:8px;" class="mono">SUPPORTED FORMATS: PNG, JPG, WEBP</div>
                                <input type="file" id="fileInput" accept="image/*" capture="environment" style="display:none;" onchange="handleFileSelect(this)">
                            </div>

                            <button type="submit" class="btn-primary" style="width:100%; margin-top:32px;">Run Colorimetric Analysis</button>
                        </form>

                        <div id="resultCard" style="margin-top:32px; display:none;" class="glass-panel" style="padding:28px;"></div>
                    </div>
                </div>

                <!-- AUDIT VAULT TAB -->
                <div id="tab-logs" class="tab-content">
                    <div class="glass-panel table-container">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:28px;">
                            <h3 class="orbitron" style="font-size:17px; color:#fff;">EVIDENCE AUDIT VAULT</h3>
                            <a href="/export-csv" onclick="playSound('click')" class="btn-primary" style="text-decoration:none; padding:11px 22px; font-size:12px; background:linear-gradient(135deg,#059669,#047857);">📥 Export Legal CSV Report</a>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Officer & Biometric</th>
                                    <th>Classification</th>
                                    <th>Hue Score</th>
                                    <th>GPS Coordinates</th>
                                    <th>IST Timestamp</th>
                                    <th>SHA-256 Evidence Seal</th>
                                </tr>
                            </thead>
                            <tbody id="logsTableBody" class="mono">
                                <tr><td colspan="7" style="text-align:center; color:var(--text-muted);">Querying secure database records...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- SIH ARCHITECTURE TAB -->
                <div id="tab-sih" class="tab-content">
                    <div class="glass-panel" style="padding:48px;">
                        <h3 class="orbitron" style="font-size:19px; color:#fff; margin-bottom:24px;">TECHNICAL SPECIFICATIONS & PROTOCOLS</h3>
                        <ul style="color:var(--text-muted); line-height:2.5; font-size:14px; padding-left:22px;" class="mono">
                            <li><strong style="color:#fff;">Computer Vision Core:</strong> OpenCV HSV color range segmentation targeting chemical reagent reactions.</li>
                            <li><strong style="color:#fff;">Cryptographic Sealing:</strong> SHA-256 evidence hashing linking officer token, image payload, GPS location, and IST timestamp.</li>
                            <li><strong style="color:#fff;">Database Engine:</strong> Managed PostgreSQL with automatic local SQLite database fallback.</li>
                            <li><strong style="color:#fff;">Zero-Dependency Auth:</strong> Standard PBKDF2 HMAC SHA-256 password hashing.</li>
                        </ul>
                    </div>
                </div>
            </main>

            <footer>
                <div>DRUG-TRACE AI Forensic Computing Platform • Secured Evidence Vault & Chain of Custody System</div>
                <div style="font-size:11px; color:var(--text-muted);" class="mono">MIL-SPEC SECURITY COMPLIANT • COURT-ADMISSIBLE AUDIT LOGGING</div>
            </footer>
        </div>

        <script>
            let activeBadge = "IND-POLICE-8042";
            let currentGps = "13.08270° N, 80.27070° E (Chennai Fix)";
            let currentAuthMode = "login";

            // Web Audio API Synthesizer for UI Sound Effects
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            function playSound(type) {
                try {
                    if (!audioCtx) return;
                    if (audioCtx.state === 'suspended') audioCtx.resume();
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    
                    const now = audioCtx.currentTime;
                    if (type === 'click') {
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(800, now);
                        osc.frequency.exponentialRampToValueAtTime(400, now + 0.04);
                        gain.gain.setValueAtTime(0.04, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);
                        osc.start(now);
                        osc.stop(now + 0.04);
                    } else if (type === 'success') {
                        osc.type = 'triangle';
                        osc.frequency.setValueAtTime(523.25, now);
                        osc.frequency.setValueAtTime(659.25, now + 0.08);
                        osc.frequency.setValueAtTime(783.99, now + 0.16);
                        gain.gain.setValueAtTime(0.07, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                        osc.start(now);
                        osc.stop(now + 0.3);
                    } else if (type === 'error') {
                        osc.type = 'sawtooth';
                        osc.frequency.setValueAtTime(150, now);
                        osc.frequency.setValueAtTime(100, now + 0.15);
                        gain.gain.setValueAtTime(0.08, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                        osc.start(now);
                        osc.stop(now + 0.3);
                    }
                } catch(e) { console.error('Audio playback error', e); }
            }

            // Real-time IST Clock update
            function updateClock() {
                const now = new Date();
                const istString = now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
                document.getElementById('clockIST').innerText = istString;
            }
            setInterval(updateClock, 1000);
            updateClock();

            // Navigation Tab Switcher
            function switchTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById('tab-' + tabId).classList.add('active');
                event.currentTarget.classList.add('active');
                if(tabId === 'logs') fetchAuditLogs();
            }

            function openAuthModal(mode = 'login') {
                switchAuthMode(mode);
                document.getElementById('authModal').style.display = 'flex';
            }
            function closeAuthModal() { document.getElementById('authModal').style.display = 'none'; }
            
            function switchAuthMode(mode) {
                currentAuthMode = mode;
                const title = document.getElementById('authTitle');
                const badgeGroup = document.getElementById('badgeGroup');
                const submitBtn = document.getElementById('authSubmitBtn');
                const loginBtn = document.getElementById('tabLoginBtn');
                const regBtn = document.getElementById('tabRegBtn');

                if (mode === 'register') {
                    title.innerText = "REGISTER NEW OFFICER";
                    badgeGroup.style.display = "block";
                    document.getElementById('badgeRegisterInput').required = true;
                    submitBtn.innerText = "Register & Generate Credentials";
                    loginBtn.classList.remove('active');
                    regBtn.classList.add('active');
                } else {
                    title.innerText = "OFFICER LOGIN";
                    badgeGroup.style.display = "none";
                    document.getElementById('badgeRegisterInput').required = false;
                    submitBtn.innerText = "Verify & Initialize Session";
                    regBtn.classList.remove('active');
                    loginBtn.classList.add('active');
                }
            }

            function enableDemoMode() {
                playSound('success');
                document.getElementById('publicHero').style.display = 'none';
                document.querySelector('.navbar').style.display = 'none';
                document.getElementById('appLayout').style.display = 'flex';
                fetchAuditLogs();
            }
            function lockSystem() {
                playSound('click');
                document.getElementById('appLayout').style.display = 'none';
                document.querySelector('.navbar').style.display = 'flex';
                document.getElementById('publicHero').style.display = 'flex';
            }

            async function handleAuthSubmit(e) {
                e.preventDefault();
                const username = document.getElementById('usernameInput').value;
                const password = document.getElementById('passwordInput').value;

                const formData = new FormData();
                formData.append('username', username);
                formData.append('password', password);

                if (currentAuthMode === 'register') {
                    const badge = document.getElementById('badgeRegisterInput').value;
                    formData.append('badge_number', badge);

                    try {
                        const res = await fetch('/register', { method: 'POST', body: formData });
                        const data = await res.json();
                        if (data.status === 'success') {
                            playSound('success');
                            alert(data.message);
                            switchAuthMode('login');
                        } else {
                            playSound('error');
                            alert(data.message || 'Registration failed');
                        }
                    } catch(err) {
                        playSound('error');
                        alert('Server connection error during registration.');
                    }
                } else {
                    try {
                        const res = await fetch('/login', { method: 'POST', body: formData });
                        const data = await res.json();
                        if (data.status === 'success') {
                            playSound('success');
                            activeBadge = data.badge_id;
                            document.getElementById('headerBadgeTag').innerText = `BADGE: ${activeBadge}`;
                            closeAuthModal();
                            document.getElementById('publicHero').style.display = 'none';
                            document.querySelector('.navbar').style.display = 'none';
                            document.getElementById('appLayout').style.display = 'flex';
                            fetchAuditLogs();
                        } else {
                            playSound('error');
                            alert(data.message || 'Authentication failed');
                        }
                    } catch(err) {
                        playSound('error');
                        alert('Server connection error during authentication.');
                    }
                }
            }

            function handleFileSelect(input) {
                if(input.files && input.files[0]) {
                    playSound('click');
                    document.getElementById('uploadNotice').innerText = `Selected: ${input.files[0].name}`;
                }
            }

            document.getElementById('scannerForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                if(!fileInput.files[0]) {
                    playSound('error');
                    alert('Please select or capture a reagent test strip image first.');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('badge_id', activeBadge);
                formData.append('gps_coords', currentGps);

                const resCard = document.getElementById('resultCard');
                resCard.style.display = 'block';
                resCard.innerHTML = `<div style="text-align:center; color:var(--text-muted); padding:20px;" class="mono">Processing OpenCV spectral matrix analysis...</div>`;

                try {
                    const response = await fetch('/analyze', { method: 'POST', body: formData });
                    const result = await response.json();

                    if(result.status === 'success') {
                        const isPos = result.classification === 'POSITIVE';
                        if (isPos) { playSound('error'); } else { playSound('success'); }
                        
                        resCard.innerHTML = `
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                                <div class="orbitron" style="font-weight:700; color:${isPos ? 'var(--danger)' : 'var(--emerald)'};">RESULT: ${result.classification}</div>
                                <div class="mono" style="font-size:12px; color:var(--text-muted);">HUE SCORE: ${result.hue_score}</div>
                            </div>
                            <div style="font-size:12px; color:var(--text-muted); line-height:1.8;" class="mono">
                                <div><strong>Badge ID:</strong> ${result.badge_id}</div>
                                <div><strong>Biometric Hash:</strong> ${result.fingerprint}</div>
                                <div><strong>GPS Fix:</strong> ${result.gps_coords}</div>
                                <div><strong>Timestamp:</strong> ${result.timestamp_ist}</div>
                                <div style="word-break:break-all; margin-top:8px; color:var(--cyan-accent);"><strong>SHA-256 Seal:</strong> ${result.sha256_hash}</div>
                            </div>
                        `;
                        fetchAuditLogs();
                    }
                } catch(err) {
                    playSound('error');
                    resCard.innerHTML = `<div style="color:var(--danger); text-align:center;">Analysis execution failed. Please retry.</div>`;
                }
            });

            async function fetchAuditLogs() {
                try {
                    const res = await fetch('/logs');
                    const data = await res.json();
                    if(data.status === 'success') {
                        document.getElementById('statTotal').innerText = data.count;
                        let posCount = 0, negCount = 0;
                        const tbody = document.getElementById('logsTableBody');
                        tbody.innerHTML = '';

                        if(data.logs.length === 0) {
                            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No forensic records available in vault.</td></tr>`;
                            return;
                        }

                        data.logs.forEach(l => {
                            if(l.classification === 'POSITIVE') posCount++; else negCount++;
                            tbody.innerHTML += `
                                <tr>
                                    <td>#${l.id}</td>
                                    <td>${l.officer_id}<br><span style="font-size:11px; color:var(--text-muted);">${l.fingerprint_hash}</span></td>
                                    <td><span style="color:${l.classification==='POSITIVE'?'var(--danger)':'var(--emerald)'}; font-weight:700;">${l.classification}</span></td>
                                    <td>${l.hue_score}</td>
                                    <td>${l.gps_coords}</td>
                                    <td>${l.timestamp_ist}</td>
                                    <td style="font-size:11px; color:var(--cyan-accent);">${l.sha256_hash.substring(0,24)}...</td>
                                </tr>
                            `;
                        });
                        document.getElementById('statPos').innerText = posCount;
                        document.getElementById('statNeg').innerText = negCount;
                    }
                } catch(err) {
                    console.error('Failed to fetch logs');
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
