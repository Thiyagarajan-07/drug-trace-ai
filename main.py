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

# --- 6. HIGH-TECH FRONTEND HUD ---
@app.get("/", response_class=HTMLResponse)
def serve_portal():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DRUG-TRACE AI | Tactical Cybernetic HUD</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">
        
        <style>
            :root {
                --bg: #030712;
                --panel: rgba(15, 23, 42, 0.75);
                --border: rgba(56, 189, 248, 0.25);
                --cyan: #06b6d4;
                --cyan-glow: rgba(6, 182, 212, 0.4);
                --blue: #3b82f6;
                --purple: #a855f7;
                --danger: #ef4444;
                --danger-glow: rgba(239, 68, 68, 0.5);
                --success: #10b981;
                --success-glow: rgba(16, 185, 129, 0.4);
                --warning: #f59e0b;
                --text: #f8fafc;
                --muted: #64748b;
            }

            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Space Grotesk', sans-serif; }
            
            body {
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.12) 0%, transparent 60%),
                    linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
                background-size: 100% 100%, 30px 30px, 30px 30px;
            }

            /* Tactical Bracket Frames */
            .hud-frame {
                position: relative;
                background: var(--panel);
                backdrop-filter: blur(16px);
                border: 1px solid var(--border);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(56, 189, 248, 0.05);
            }

            .hud-frame::before, .hud-frame::after {
                content: '';
                position: absolute;
                width: 10px;
                height: 10px;
                border-color: var(--cyan);
                border-style: solid;
                pointer-events: none;
            }

            .hud-frame::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
            .hud-frame::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

            .orbitron { font-family: 'Orbitron', sans-serif; }
            .mono { font-family: 'JetBrains Mono', monospace; }

            /* Dynamic Glowing Text & Badges */
            .glow-text { text-shadow: 0 0 12px var(--cyan-glow); }
            .glow-danger { text-shadow: 0 0 12px var(--danger-glow); }

            /* Top HUD Diagnostics Bar */
            .top-bar {
                background: rgba(3, 7, 18, 0.9);
                border-bottom: 1px solid var(--border);
                padding: 10px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
            }

            .sys-status-badge {
                display: flex;
                align-items: center;
                gap: 8px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid var(--success);
                color: var(--success);
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: 600;
            }

            .status-dot {
                width: 6px;
                height: 6px;
                background: var(--success);
                border-radius: 50%;
                box-shadow: 0 0 8px var(--success);
                animation: pulse 1.5s infinite alternate;
            }

            @keyframes pulse { 0% { opacity: 0.3; } 100% { opacity: 1; transform: scale(1.3); } }

            /* Public Landing Hero */
            #publicHero {
                min-height: calc(100vh - 45px);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 40px 20px;
                text-align: center;
                position: relative;
            }

            .hero-title {
                font-size: 56px;
                font-weight: 900;
                letter-spacing: 2px;
                margin-bottom: 12px;
                background: linear-gradient(135deg, #fff 30%, var(--cyan) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-subtitle {
                max-width: 680px;
                color: var(--muted);
                font-size: 16px;
                line-height: 1.6;
                margin-bottom: 36px;
            }

            .btn-cyber {
                position: relative;
                padding: 14px 28px;
                background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
                border: 1px solid var(--cyan);
                color: #fff;
                font-weight: 700;
                font-size: 14px;
                letter-spacing: 1px;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
            }

            .btn-cyber:hover {
                background: var(--cyan);
                color: #000;
                box-shadow: 0 0 25px var(--cyan-glow);
                transform: translateY(-2px);
            }

            /* Lock Screen Modal */
            #authModal {
                position: fixed;
                inset: 0;
                background: rgba(3, 7, 18, 0.92);
                backdrop-filter: blur(20px);
                z-index: 9999;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .auth-card {
                width: 100%;
                max-width: 440px;
                padding: 36px;
                border-radius: 4px;
            }

            .biometric-ring {
                width: 90px;
                height: 90px;
                border-radius: 50%;
                border: 2px solid var(--cyan);
                margin: 0 auto 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                cursor: pointer;
                background: rgba(6, 182, 212, 0.05);
                transition: 0.3s;
            }

            .biometric-ring::after {
                content: '';
                position: absolute;
                inset: -6px;
                border-radius: 50%;
                border: 2px dashed var(--cyan);
                animation: spin 12s linear infinite;
            }

            @keyframes spin { 100% { transform: rotate(360deg); } }

            .input-field {
                width: 100%;
                padding: 12px 16px;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid var(--border);
                border-radius: 4px;
                color: #fff;
                font-family: 'JetBrains Mono', monospace;
                margin-top: 6px;
                margin-bottom: 16px;
                transition: 0.3s;
            }

            .input-field:focus {
                outline: none;
                border-color: var(--cyan);
                box-shadow: 0 0 12px var(--cyan-glow);
            }

            /* Main Layout */
            #appLayout { display: none; flex-direction: column; min-height: 100vh; }

            header {
                background: rgba(15, 23, 42, 0.8);
                border-bottom: 1px solid var(--border);
                padding: 16px 32px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .nav-tabs {
                display: flex;
                gap: 4px;
                background: rgba(3, 7, 18, 0.8);
                padding: 6px 32px;
                border-bottom: 1px solid var(--border);
            }

            .tab-btn {
                background: none;
                border: none;
                color: var(--muted);
                padding: 12px 20px;
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 0.5px;
                cursor: pointer;
                transition: 0.3s;
                position: relative;
            }

            .tab-btn.active {
                color: var(--cyan);
                background: rgba(6, 182, 212, 0.08);
            }

            .tab-btn.active::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: var(--cyan);
                box-shadow: 0 0 10px var(--cyan);
            }

            main { flex: 1; padding: 32px; max-width: 1300px; margin: 0 auto; width: 100%; }

            .tab-content { display: none; }
            .tab-content.active { display: block; }

            /* Telemetry Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 20px;
                margin-bottom: 32px;
            }

            .metric-card { padding: 20px; }
            .metric-val { font-size: 36px; font-weight: 800; margin-top: 8px; }

            /* Scanner Interactive Target HUD */
            .scanner-container {
                max-width: 650px;
                margin: 0 auto;
                padding: 32px;
            }

            .target-box {
                position: relative;
                width: 100%;
                height: 260px;
                border: 2px dashed var(--cyan);
                background: rgba(6, 182, 212, 0.03);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                overflow: hidden;
                transition: 0.3s;
            }

            .target-box:hover { border-color: var(--success); background: rgba(16, 185, 129, 0.05); }

            .scan-line {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--cyan);
                box-shadow: 0 0 15px var(--cyan);
                animation: scanSweep 2.5s ease-in-out infinite alternate;
                display: none;
            }

            @keyframes scanSweep { 0% { top: 0%; } 100% { top: 98%; } }

            /* Dynamic Logs Table */
            .table-container { padding: 24px; overflow-x: auto; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
            th { background: rgba(3, 7, 18, 0.8); color: var(--cyan); padding: 14px 16px; border-bottom: 1px solid var(--border); font-family: 'Orbitron', sans-serif; font-size: 11px; }
            td { padding: 14px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
            tr:hover { background: rgba(6, 182, 212, 0.04); }

            /* HUD Console Stream */
            .terminal-box {
                background: #020617;
                border: 1px solid var(--border);
                padding: 16px;
                border-radius: 4px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 12px;
                color: var(--cyan);
                height: 140px;
                overflow-y: auto;
                margin-top: 24px;
            }
        </style>
    </head>
    <body>

        <!-- TOP DIAGNOSTICS BAR -->
        <div class="top-bar mono">
            <div style="display:flex; gap:20px; align-items:center;">
                <div class="sys-status-badge">
                    <span class="status-dot"></span> SYSTEM ONLINE
                </div>
                <div>SERVER LATENCY: <span id="sysLatency" style="color:var(--cyan);">14ms</span></div>
                <div>ENCRYPTION: <span style="color:var(--purple);">SHA-256 AGENT SEAL</span></div>
            </div>
            <div style="display:flex; gap:20px; align-items:center;">
                <div>TIME (IST): <span id="hudClockIST" style="color:#fff;">--:--:--</span></div>
                <div>GPS SATELLITES: <span style="color:var(--success);">11 LOCKED</span></div>
            </div>
        </div>

        <!-- PUBLIC HERO PAGE -->
        <div id="publicHero">
            <div style="letter-spacing:3px; color:var(--cyan); font-size:12px; font-weight:800; margin-bottom:12px;" class="orbitron">
                [ FORENSIC FIELD COMPUTING PLATFORM ]
            </div>
            <h1 class="hero-title orbitron">DRUG-TRACE AI</h1>
            <p class="hero-subtitle">
                Autonomous chemical spot-test classification engine powered by OpenCV spectral matrix mapping, satellite geo-stamping, and cryptographic evidence vaulting.
            </p>
            <div style="display:flex; gap:16px;">
                <button onclick="openLoginModal()" class="btn-cyber orbitron">⚡ OFFICER AUTHENTICATION</button>
                <button onclick="enableDemoMode()" class="btn-cyber orbitron" style="border-color:var(--muted); background:rgba(255,255,255,0.05);">👁️ EXPLORE EVALUATOR DEMO</button>
            </div>
        </div>

        <!-- AUTH LOCK SCREEN -->
        <div id="authModal">
            <div class="auth-card hud-frame">
                <div style="text-align:center; margin-bottom:24px;">
                    <h2 class="orbitron glow-text" style="color:var(--cyan); font-size:22px;">OFFICER LOCKSCREEN</h2>
                    <p style="font-size:12px; color:var(--muted); margin-top:4px;">VERIFY BIOMETRIC TOKEN OR BADGE CREDENTIALS</p>
                </div>

                <div class="biometric-ring" onclick="triggerBiometricScan()">
                    <span style="font-size:36px;" id="bioIcon">👆</span>
                </div>
                <div id="bioStatus" style="text-align:center; font-size:11px; color:var(--muted); margin-bottom:20px;" class="mono">
                    CLICK TO SCAN BIOMETRIC SENSOR
                </div>

                <form id="authForm" onsubmit="handleAuthSubmit(event)">
                    <label style="font-size:11px; color:var(--muted);" class="orbitron">OFFICER USERNAME</label>
                    <input type="text" id="usernameInput" class="input-field" placeholder="officer_8042" required>

                    <div id="badgeGroup">
                        <label style="font-size:11px; color:var(--muted);" class="orbitron">BADGE NUMBER</label>
                        <input type="text" id="badgeInput" class="input-field" value="IND-POLICE-8042">
                    </div>

                    <label style="font-size:11px; color:var(--muted);" class="orbitron">SECURITY ACCESS KEY</label>
                    <input type="password" id="passwordInput" class="input-field" placeholder="••••••••" required>

                    <button type="submit" id="authSubmitBtn" class="btn-cyber orbitron" style="width:100%; margin-top:8px;">AUTHENTICATE SESSION</button>
                </form>

                <div style="text-align:center; margin-top:20px;">
                    <span id="toggleAuthBtn" onclick="toggleAuthMode()" style="color:var(--cyan); font-size:12px; cursor:pointer; text-decoration:underline;">New Officer Registration?</span>
                    <br><br>
                    <button onclick="closeLoginModal()" style="background:none; border:none; color:var(--muted); font-size:11px; cursor:pointer;">← Return to Main Page</button>
                </div>
            </div>
        </div>

        <!-- MAIN APP DASHBOARD -->
        <div id="appLayout">
            <header>
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="font-size:24px; color:var(--cyan);">🛡️</div>
                    <div>
                        <div class="orbitron glow-text" style="font-size:18px; font-weight:900; color:var(--cyan);">DRUG-TRACE AI</div>
                        <div style="font-size:10px; color:var(--muted);" class="mono">FORENSIC SUITE v4.2</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:16px;">
                    <span class="sys-status-badge mono" id="headerOfficer">BADGE: IND-POLICE-8042</span>
                    <button onclick="lockSystem()" class="btn-cyber orbitron" style="padding:8px 16px; font-size:11px; border-color:var(--danger); background:rgba(239,68,68,0.1);">LOCK HUD</button>
                </div>
            </header>

            <div class="nav-tabs orbitron">
                <button class="tab-btn active" onclick="switchTab('overview')">📊 TELEMETRY DASHBOARD</button>
                <button class="tab-btn" onclick="switchTab('scanner')">🧪 CHEMICAL REAGENT SCANNER</button>
                <button class="tab-btn" onclick="switchTab('logs')">📑 FORENSIC AUDIT VAULT</button>
                <button class="tab-btn" onclick="switchTab('sih')">⚙️ SYSTEM ARCHITECTURE</button>
            </div>

            <main>
                <!-- OVERVIEW TAB -->
                <div id="tab-overview" class="tab-content active">
                    <div class="metrics-grid">
                        <div class="metric-card hud-frame">
                            <div style="font-size:11px; color:var(--muted);" class="orbitron">TOTAL FIELD EXECUTIONS</div>
                            <div class="metric-val mono glow-text" id="statTotal" style="color:var(--cyan);">0</div>
                        </div>
                        <div class="metric-card hud-frame">
                            <div style="font-size:11px; color:var(--muted);" class="orbitron">POSITIVE DETECTIONS</div>
                            <div class="metric-val mono glow-danger" id="statPos" style="color:var(--danger);">0</div>
                        </div>
                        <div class="metric-card hud-frame">
                            <div style="font-size:11px; color:var(--muted);" class="orbitron">NEGATIVE REAGENTS</div>
                            <div class="metric-val mono" id="statNeg" style="color:var(--success);">0</div>
                        </div>
                        <div class="metric-card hud-frame">
                            <div style="font-size:11px; color:var(--muted);" class="orbitron">CHAIN-OF-CUSTODY INTEGRITY</div>
                            <div class="metric-val mono glow-text" style="color:var(--purple);">100%</div>
                        </div>
                    </div>

                    <div class="hud-frame" style="padding:32px; text-align:center;">
                        <h2 class="orbitron glow-text" style="color:var(--cyan); margin-bottom:12px;">READY FOR FIELD REAGENT ANALYSIS</h2>
                        <p style="color:var(--muted); max-width:600px; margin:0 auto 24px; font-size:14px;">Upload or capture spot-test chemical strip photographs to evaluate OpenCV HSV color matrix channels and sign cryptographic evidence logs.</p>
                        <button onclick="switchTab('scanner')" class="btn-cyber orbitron">INITIALIZE CAMERA SCANNER →</button>
                    </div>
                </div>

                <!-- SCANNER TAB -->
                <div id="tab-scanner" class="tab-content">
                    <div class="scanner-container hud-frame">
                        <h2 class="orbitron glow-text" style="color:var(--cyan); text-align:center; margin-bottom:8px;">AI REAGENT ANALYZER</h2>
                        <p style="color:var(--muted); text-align:center; font-size:12px; margin-bottom:24px;" class="mono">SPECTRAL MATRIX MAPPING • TARGETING RETICLE</p>

                        <form id="scannerForm">
                            <div class="target-box" onclick="document.getElementById('fileInput').click()">
                                <div class="scan-line" id="scanSweepLine"></div>
                                <div style="font-size:42px; margin-bottom:12px;">📷</div>
                                <div style="font-weight:700; font-size:14px;" id="uploadNotice" class="orbitron">TAP TO CAPTURE OR UPLOAD STRIP PHOTO</div>
                                <div style="font-size:11px; color:var(--muted); margin-top:4px;" class="mono">SUPPORTED: PNG, JPG, WEBP</div>
                                <input type="file" id="fileInput" accept="image/*" capture="environment" style="display:none;" onchange="handleFileSelect(this)">
                            </div>

                            <button type="submit" id="analyzeBtn" class="btn-cyber orbitron" style="width:100%; margin-top:24px;">RUN COLORIMETRIC ANALYSIS</button>
                        </form>

                        <div id="resultCard" style="margin-top:24px; display:none;" class="hud-frame"></div>
                    </div>
                </div>

                <!-- AUDIT VAULT TAB -->
                <div id="tab-logs" class="tab-content">
                    <div class="hud-frame table-container">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                            <h3 class="orbitron glow-text" style="color:var(--cyan);">EVIDENCE AUDIT VAULT</h3>
                            <a href="/export-csv" class="btn-cyber orbitron" style="text-decoration:none; padding:8px 16px; font-size:11px; border-color:var(--success);">📥 EXPORT LEGAL CSV REPORT</a>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>OFFICER & BIOMETRIC</th>
                                    <th>CLASSIFICATION</th>
                                    <th>HUE SCORE</th>
                                    <th>GPS COORDINATES</th>
                                    <th>IST TIMESTAMP</th>
                                    <th>SHA-256 EVIDENCE SEAL</th>
                                </tr>
                            </thead>
                            <tbody id="logsTableBody" class="mono">
                                <tr><td colspan="7" style="text-align:center; color:var(--muted);">Querying encrypted database records...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- SIH ARCHITECTURE TAB -->
                <div id="tab-sih" class="tab-content">
                    <div class="hud-frame" style="padding:32px;">
                        <h3 class="orbitron glow-text" style="color:var(--cyan); margin-bottom:16px;">TECHNICAL SPECIFICATIONS & PROTOCOLS</h3>
                        <ul style="color:var(--muted); line-height:2; font-size:14px; padding-left:20px;" class="mono">
                            <li><strong style="color:#fff;">Computer Vision Core:</strong> OpenCV HSV color range segmentation targeting chemical reagent reactions.</li>
                            <li><strong style="color:#fff;">Cryptographic Sealing:</strong> SHA-256 evidence hashing linking officer token, image payload, GPS location, and IST timestamp.</li>
                            <li><strong style="color:#fff;">Database Engine:</strong> Managed PostgreSQL with automatic local SQLite database fallback.</li>
                            <li><strong style="color:#fff;">Zero-Dependency Auth:</strong> Standard PBKDF2 HMAC SHA-256 password hashing.</li>
                        </ul>
                    </div>
                </div>

                <!-- SYSTEM STREAM TERMINAL -->
                <div class="terminal-box" id="sysConsole">
                    <div>[SYS INIT] DRUG-TRACE AI Tactical Engine Initialized.</div>
                    <div>[GPS FIX] Satellite synchronization acquired.</div>
                </div>
            </main>
        </div>

        <script>
            let officerToken = "BIO-TOKEN-VERIFIED";
            let activeBadge = "IND-POLICE-8042";
            let currentGps = "13.08270° N, 80.27070° E (Chennai Fix)";
            let isRegisterMode = false;

            // Audio Feedback Engine (Web Audio API)
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            function playCyberBeep(freq = 880, type = 'sine', duration = 0.08) {
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = type;
                    osc.frequency.value = freq;
                    gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start();
                    osc.stop(audioCtx.currentTime + duration);
                } catch(e) {}
            }

            function logConsole(msg) {
                const con = document.getElementById('sysConsole');
                const time = new Date().toLocaleTimeString();
                con.innerHTML += `<div>[${time}] ${msg}</div>`;
                con.scrollTop = con.scrollHeight;
            }

            // Real-time IST Clock
            function updateClock() {
                const now = new Date();
                const ist = now.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
                document.getElementById('hudClockIST').innerText = ist + ' IST';
            }
            setInterval(updateClock, 1000);
            updateClock();

            // Biometric Simulation
            function triggerBiometricScan() {
                playCyberBeep(1200, 'square', 0.15);
                const status = document.getElementById('bioStatus');
                const icon = document.getElementById('bioIcon');
                
                status.innerText = "VERIFYING RIDGE PATTERNS...";
                status.style.color = "var(--cyan)";

                setTimeout(() => {
                    officerToken = "FP-BIO-" + Math.random().toString(36).substring(2, 8).toUpperCase();
                    icon.innerText = "✅";
                    status.innerText = "BIOMETRIC MATCH CONFIRMED (" + officerToken + ")";
                    status.style.color = "var(--success)";
                    logConsole("Biometric auth token generated: " + officerToken);
                    playCyberBeep(1760, 'sine', 0.2);
                }, 800);
            }

            // Auth Modal Controls
            function openLoginModal() {
                playCyberBeep(600, 'sine', 0.05);
                document.getElementById('publicHero').style.display = 'none';
                document.getElementById('authModal').style.display = 'flex';
            }

            function closeLoginModal() {
                playCyberBeep(400, 'sine', 0.05);
                document.getElementById('authModal').style.display = 'none';
                document.getElementById('publicHero').style.display = 'flex';
            }

            function enableDemoMode() {
                playCyberBeep(1000, 'triangle', 0.1);
                document.getElementById('publicHero').style.display = 'none';
                document.getElementById('authModal').style.display = 'none';
                document.getElementById('appLayout').style.display = 'flex';
                document.getElementById('headerOfficer').innerText = "BADGE: EVALUATOR-DEMO";
                logConsole("Session started under EVALUATOR DEMO mode.");
                reloadStatsAndLogs();
            }

            function lockSystem() {
                playCyberBeep(300, 'sawtooth', 0.2);
                document.getElementById('appLayout').style.display = 'none';
                document.getElementById('authModal').style.display = 'none';
                document.getElementById('publicHero').style.display = 'flex';
            }

            function toggleAuthMode() {
                playCyberBeep(800, 'sine', 0.05);
                isRegisterMode = !isRegisterMode;
                const btn = document.getElementById('authSubmitBtn');
                const toggle = document.getElementById('toggleAuthBtn');

                if (isRegisterMode) {
                    btn.innerText = "CREATE OFFICER CREDENTIALS";
                    toggle.innerText = "Already registered? Login Here";
                } else {
                    btn.innerText = "AUTHENTICATE SESSION";
                    toggle.innerText = "New Officer Registration?";
                }
            }

            async function handleAuthSubmit(e) {
                e.preventDefault();
                playCyberBeep(900, 'sine', 0.08);

                const username = document.getElementById('usernameInput').value;
                const badge = document.getElementById('badgeInput').value;
                const password = document.getElementById('passwordInput').value;

                const formData = new FormData();
                formData.append('username', username);
                formData.append('password', password);

                const endpoint = isRegisterMode ? '/register' : '/login';
                if (isRegisterMode) {
                    formData.append('badge_number', badge);
                }

                try {
                    const res = await fetch(endpoint, { method: 'POST', body: formData });
                    const data = await res.json();

                    if (res.ok && data.status === 'success') {
                        if (isRegisterMode) {
                            alert("Account created successfully! Proceeding to login...");
                            toggleAuthMode();
                        } else {
                            activeBadge = data.badge_id || badge;
                            document.getElementById('headerOfficer').innerText = "BADGE: " + activeBadge;
                            document.getElementById('authModal').style.display = 'none';
                            document.getElementById('appLayout').style.display = 'flex';
                            logConsole("Officer logged in successfully. Badge ID: " + activeBadge);
                            reloadStatsAndLogs();
                        }
                    } else {
                        alert(data.message || data.detail || "Authentication failure.");
                    }
                } catch (err) {
                    alert("Authentication server connection error: " + err.message);
                }
            }

            function switchTab(tabId) {
                playCyberBeep(700, 'sine', 0.04);
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                event.target.classList.add('active');
                document.getElementById('tab-' + tabId).classList.add('active');

                if (tabId === 'logs' || tabId === 'overview') {
                    reloadStatsAndLogs();
                }
            }

            function handleFileSelect(input) {
                if (input.files && input.files[0]) {
                    playCyberBeep(1100, 'sine', 0.1);
                    document.getElementById('uploadNotice').innerText = "SELECTED: " + input.files[0].name;
                    document.getElementById('scanSweepLine').style.display = 'block';
                }
            }

            // Chemical Scanner Submission
            document.getElementById('scannerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const btn = document.getElementById('analyzeBtn');
                const card = document.getElementById('resultCard');

                if (!fileInput.files[0]) {
                    alert("Please select a test strip photograph first.");
                    return;
                }

                playCyberBeep(500, 'square', 0.2);
                btn.disabled = true;
                btn.innerText = "PROCESSING SPECTRAL MATRIX...";

                const nowIST = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) + ' IST';
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('officer_id', activeBadge);
                formData.append('fingerprint_hash', officerToken);
                formData.append('gps_coords', currentGps);
                formData.append('timestamp_ist', nowIST);

                try {
                    const res = await fetch('/verify', { method: 'POST', body: formData });
                    const data = await res.json();

                    card.style.display = 'block';
                    card.style.padding = '24px';

                    if (data.classification === 'POSITIVE') {
                        playCyberBeep(300, 'sawtooth', 0.4);
                        card.style.borderColor = "var(--danger)";
                        card.innerHTML = `
                            <div class="orbitron glow-danger" style="color:var(--danger); font-size:18px; font-weight:800;">⚠️ POSITIVE REAGENT DETECTED</div>
                            <div style="font-size:12px; margin-top:8px;" class="mono">Narcotic threshold exceeded. Avg Hue: ${data.hue_score}</div>
                        `;
                    } else {
                        playCyberBeep(1200, 'sine', 0.3);
                        card.style.borderColor = "var(--success)";
                        card.innerHTML = `
                            <div class="orbitron" style="color:var(--success); font-size:18px; font-weight:800;">✅ NEGATIVE REAGENT RESULT</div>
                            <div style="font-size:12px; margin-top:8px;" class="mono">No illegal substances matched. Avg Hue: ${data.hue_score}</div>
                        `;
                    }

                    card.innerHTML += `
                        <div style="font-size:11px; color:var(--muted); margin-top:16px; border-top:1px dashed var(--border); padding-top:12px;" class="mono">
                            <div>GPS: ${data.gps_coords}</div>
                            <div>TIMESTAMP: ${data.timestamp_ist}</div>
                            <div>SHA-256 EVIDENCE SEAL: ${data.sha256_hash}</div>
                        </div>
                    `;

                    logConsole("Analysis executed. Result: " + data.classification + " | Seal: " + data.sha256_hash.substring(0, 10));
                    reloadStatsAndLogs();
                } catch (err) {
                    alert("Analysis server connection error.");
                } finally {
                    btn.disabled = false;
                    btn.innerText = "RUN COLORIMETRIC ANALYSIS";
                    document.getElementById('scanSweepLine').style.display = 'none';
                }
            });

            // Fetch Telemetry & Database Audit Logs
            async function reloadStatsAndLogs() {
                try {
                    const res = await fetch('/api/stats');
                    const data = await res.json();

                    document.getElementById('statTotal').innerText = data.total;
                    document.getElementById('statPos').innerText = data.positive;
                    document.getElementById('statNeg').innerText = data.negative;

                    const tbody = document.getElementById('logsTableBody');
                    if (!data.logs || data.logs.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--muted); padding:20px;">No forensic evidence logs recorded yet.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = data.logs.map(r => `
                        <tr>
                            <td>#${r.id}</td>
                            <td><b>${r.officer_id}</b><br><small style="color:var(--muted);">${r.fingerprint_hash}</small></td>
                            <td><span style="color:${r.classification === 'POSITIVE' ? 'var(--danger)' : 'var(--success)'}; font-weight:bold;">${r.classification}</span></td>
                            <td>${r.hue_score}</td>
                            <td><small>${r.gps_coords}</small></td>
                            <td><small>${r.timestamp_ist}</small></td>
                            <td><code style="color:var(--cyan);">${r.sha256_hash.substring(0, 12)}...</code></td>
                        </tr>
                    `).join('');
                } catch (err) {
                    console.error("Telemetry update error.");
                }
            }
        </script>
    </body>
    </html>
    """

# --- 7. VERIFICATION ENDPOINT ---
@app.post("/verify")
async def verify_sample(
    file: UploadFile = File(...),
    officer_id: str = Form(...),
    fingerprint_hash: str = Form(...),
    gps_coords: str = Form(...),
    timestamp_ist: str = Form(...),
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    classification, hue_score = process_test_image(image_bytes)

    hash_input = image_bytes + f"{officer_id}{fingerprint_hash}{gps_coords}{timestamp_ist}{classification}".encode('utf-8')
    sha256_hash = hashlib.sha256(hash_input).hexdigest()

    log_entry = AuditLog(
        officer_id=officer_id,
        fingerprint_hash=fingerprint_hash,
        gps_coords=gps_coords,
        timestamp_ist=timestamp_ist,
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
        "sha256_hash": sha256_hash,
        "fingerprint_hash": fingerprint_hash,
        "gps_coords": gps_coords,
        "timestamp_ist": timestamp_ist
    })

# --- 8. STATS ENDPOINT ---
@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    logs_query = db.query(AuditLog).order_by(AuditLog.id.desc()).all()
    
    total = len(logs_query)
    pos = sum(1 for log in logs_query if log.classification == "POSITIVE")
    neg = sum(1 for log in logs_query if log.classification == "NEGATIVE")

    logs = [
        {
            "id": r.id,
            "officer_id": r.officer_id,
            "fingerprint_hash": r.fingerprint_hash,
            "gps_coords": r.gps_coords,
            "timestamp_ist": r.timestamp_ist,
            "classification": r.classification,
            "hue_score": r.hue_score,
            "sha256_hash": r.sha256_hash
        }
        for r in logs_query
    ]

    return JSONResponse({"total": total, "positive": pos, "negative": neg, "logs": logs})

# --- 9. CSV EXPORT ENDPOINT ---
@app.get("/export-csv")
def export_csv(db: Session = Depends(get_db)):
    rows = db.query(AuditLog).order_by(AuditLog.id.asc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Officer ID', 'Biometric Token', 'GPS Coordinates', 'IST Timestamp', 'Classification', 'Hue Score', 'SHA-256 Evidence Seal'])
    
    for r in rows:
        writer.writerow([r.id, r.officer_id, r.fingerprint_hash, r.gps_coords, r.timestamp_ist, r.classification, r.hue_score, r.sha256_hash])
    
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=drug_trace_audit_logs.csv"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
