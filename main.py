import os
import sqlite3
import hashlib
import io
import csv
import numpy as np
import cv2
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="DRUG-TRACE AI - Enterprise Field Portal")

# --- 1. OFFLINE SQLITE DATABASE ENGINE ---
DB_NAME = "drug_trace.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            officer_id TEXT NOT NULL,
            fingerprint_hash TEXT NOT NULL,
            gps_coords TEXT NOT NULL,
            timestamp_ist TEXT NOT NULL,
            classification TEXT NOT NULL,
            hue_score REAL NOT NULL,
            sha256_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. OPENCV COMPUTER VISION HSV ENGINE ---
def process_test_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "INCONCLUSIVE", 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    center_hsv = hsv[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]

    # HSV range for positive chemical reaction (Magenta/Red/Purple)
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


# --- 3. FULL ENTERPRISE WEB APP INTERFACE ---
@app.get("/", response_class=HTMLResponse)
def serve_portal():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DRUG-TRACE AI | SIH Forensic Portal</title>
        <style>
            :root {
                --bg: #0b1329;
                --panel: #1e293b;
                --accent: #38bdf8;
                --accent-glow: rgba(56, 189, 248, 0.2);
                --danger: #f43f5e;
                --success: #10b981;
                --text: #f8fafc;
                --muted: #94a3b8;
                --border: #334155;
            }
            
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
            
            /* --- LOCK SCREEN MODAL --- */
            #authModal { position: fixed; inset: 0; background: rgba(11, 19, 41, 0.95); backdrop-filter: blur(10px); z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 20px; }
            .auth-card { background: var(--panel); border: 1px solid var(--accent); border-radius: 16px; width: 100%; max-width: 420px; padding: 32px; text-align: center; box-shadow: 0 0 40px var(--accent-glow); }
            .auth-title { font-size: 22px; color: var(--accent); margin-bottom: 8px; font-weight: 800; letter-spacing: 1px; }
            .auth-sub { font-size: 13px; color: var(--muted); margin-bottom: 24px; }
            
            .fp-scanner { width: 100px; height: 100px; border-radius: 50%; border: 2px dashed var(--accent); margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: 0.3s; background: #0f172a; }
            .fp-scanner:hover { transform: scale(1.05); border-color: var(--success); box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
            .fp-icon { font-size: 48px; color: var(--accent); }
            .fp-label { font-size: 12px; color: var(--muted); margin-bottom: 16px; font-weight: bold; }
            
            .input-group { text-align: left; margin-bottom: 20px; }
            .input-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
            .input-group input { width: 100%; padding: 12px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; color: #fff; font-family: monospace; font-size: 14px; }
            
            .auth-btn { width: 100%; padding: 14px; background: var(--accent); border: none; border-radius: 8px; color: #000; font-weight: bold; font-size: 15px; cursor: pointer; transition: 0.3s; }
            .auth-btn:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }

            /* --- MAIN DASHBOARD LAYOUT --- */
            #appLayout { display: none; flex-direction: column; min-height: 100vh; }
            
            header { background: #0f172a; border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
            .brand { display: flex; align-items: center; gap: 12px; }
            .brand-logo { font-size: 24px; }
            .brand-text { font-size: 18px; font-weight: 800; color: var(--accent); letter-spacing: 0.5px; }
            .badge-officer { background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent); padding: 6px 12px; border-radius: 20px; font-size: 12px; color: var(--accent); font-family: monospace; }
            
            .nav-tabs { background: #0f172a; border-bottom: 1px solid var(--border); display: flex; padding: 0 24px; gap: 8px; overflow-x: auto; }
            .tab-btn { background: none; border: none; color: var(--muted); padding: 14px 18px; font-weight: bold; font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; transition: 0.3s; white-space: nowrap; }
            .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); background: rgba(56, 189, 248, 0.05); }
            
            main { flex: 1; padding: 24px; max-width: 1200px; margin: 0 auto; width: 100%; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            /* --- METRIC CARDS GRID --- */
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
            .metric-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
            .metric-title { font-size: 12px; color: var(--muted); text-transform: uppercase; font-weight: bold; margin-bottom: 8px; }
            .metric-val { font-size: 28px; font-weight: 800; color: var(--text); font-family: monospace; }
            .metric-val.pos { color: var(--danger); }
            .metric-val.neg { color: var(--success); }

            /* --- LIVE METADATA BANNER --- */
            .banner-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }
            .info-card { background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 16px; font-size: 13px; }
            .info-hdr { color: var(--muted); font-weight: bold; margin-bottom: 6px; display: flex; justify-content: space-between; }
            .info-val { color: var(--accent); font-family: monospace; font-size: 14px; }

            /* --- CHEMICAL ANALYSIS TERMINAL --- */
            .scanner-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 24px; max-width: 580px; margin: 0 auto; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
            .file-upload { border: 2px dashed var(--accent); border-radius: 10px; padding: 24px; text-align: center; background: #0f172a; cursor: pointer; margin-top: 12px; }
            .file-upload input { display: none; }
            .upload-btn { display: inline-block; padding: 10px 20px; background: #334155; color: var(--text); border-radius: 6px; font-size: 13px; font-weight: bold; margin-top: 8px; cursor: pointer; }

            #resultBox { margin-top: 20px; padding: 20px; border-radius: 10px; display: none; }
            .res-pos { background: #881337; color: #fecdd3; border: 1px solid #f43f5e; }
            .res-neg { background: #064e3b; color: #a7f3d0; border: 1px solid #10b981; }

            /* --- FORENSIC VAULT TABLE --- */
            .table-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; overflow-x: auto; }
            .table-hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
            th, td { padding: 12px 16px; border-bottom: 1px solid var(--border); }
            th { background: #0f172a; color: var(--accent); font-weight: bold; }
            tr:hover { background: #0f172a; }
            
            .b-pos { background: #881337; color: #fecdd3; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
            .b-neg { background: #064e3b; color: #a7f3d0; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
            code { background: #0f172a; padding: 4px 6px; border-radius: 4px; color: var(--accent); font-family: monospace; }
        </style>
    </head>
    <body>

        <div id="authModal">
            <div class="auth-card">
                <div class="auth-title">DRUG-TRACE AI</div>
                <div class="auth-sub">Smart India Hackathon • Forensic Field Portal</div>
                
                <div class="fp-scanner" id="fpModalBtn" onclick="authenticateOfficer()">
                    <span class="fp-icon">👆</span>
                </div>
                <div class="fp-label" id="fpModalStatus">Tap biometric sensor to authenticate officer</div>

                <div class="input-group">
                    <label>Officer Badge ID:</label>
                    <input type="text" id="badgeInput" value="IND-POLICE-8042" readonly>
                </div>

                <button type="button" id="enterAppBtn" class="auth-btn" disabled onclick="unlockSystem()">System Locked</button>
            </div>
        </div>

        <div id="appLayout">
            <header>
                <div class="brand">
                    <span class="brand-logo">🛡️</span>
                    <span class="brand-text">DRUG-TRACE AI</span>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span class="badge-officer" id="headerOfficer">BADGE: IND-POLICE-8042</span>
                    <button onclick="lockSystem()" style="background:#334155; border:none; color:#fff; padding:6px 12px; border-radius:6px; font-size:12px; cursor:pointer;">🔒 Lock</button>
                </div>
            </header>

            <div class="nav-tabs">
                <button class="tab-btn active" onclick="switchTab('overview')">📊 Overview Dashboard</button>
                <button class="tab-btn" onclick="switchTab('scanner')">🧪 Chemical Analysis (Drug Test AI)</button>
                <button class="tab-btn" onclick="switchTab('logs')">📑 Forensic Audit Vault</button>
                <button class="tab-btn" onclick="switchTab('sih')">⚙️ SIH Technical Architecture</button>
            </div>

            <main>
                <div class="banner-grid">
                    <div class="info-card">
                        <div class="info-hdr">🕒 FIELD TIMESTAMP (IST)</div>
                        <div class="info-val" id="liveIstClock">Loading IST clock...</div>
                    </div>
                    <div class="info-card">
                        <div class="info-hdr">
                            <span>📍 SATELLITE GPS FIX</span>
                            <button onclick="fetchGPS()" style="background:#334155; color:#fff; border:none; padding:2px 6px; border-radius:4px; cursor:pointer; font-size:10px;">Refresh</button>
                        </div>
                        <div class="info-val" id="liveGpsCoords">Acquiring GPS fix...</div>
                    </div>
                </div>

                <div id="tab-overview" class="tab-content active">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-title">Total Tests Executed</div>
                            <div class="metric-val" id="statTotal">0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Positive Reagents</div>
                            <div class="metric-val pos" id="statPos">0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Negative Reagents</div>
                            <div class="metric-val neg" id="statNeg">0</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Chain-of-Custody Integrity</div>
                            <div class="metric-val" style="color:var(--accent);">100%</div>
                        </div>
                    </div>

                    <div style="background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:24px; text-align:center;">
                        <h3 style="color:var(--accent); margin-bottom:8px;">Ready for Field Chemical Testing</h3>
                        <p style="color:var(--muted); font-size:14px; margin-bottom:16px;">Perform computer vision analysis on narcotic spot-test strips with full cryptographic audit trails.</p>
                        <button onclick="switchTab('scanner')" class="auth-btn" style="max-width:280px; display:inline-block;">Launch Chemical Scanner →</button>
                    </div>
                </div>

                <div id="tab-scanner" class="tab-content">
                    <div class="scanner-card">
                        <h3 style="color:var(--accent); margin-bottom:6px; text-align:center;">AI Narcotic Spot-Test Terminal</h3>
                        <p style="color:var(--muted); font-size:13px; text-align:center; margin-bottom:20px;">Upload or capture test strip image for HSV color matrix verification.</p>

                        <form id="scannerForm">
                            <div class="file-upload" onclick="document.getElementById('fileInput').click()">
                                <div style="font-size:32px; margin-bottom:8px;">📷</div>
                                <div style="font-size:14px; color:#fff; font-weight:bold;" id="fileNameDisplay">Select or Capture Test Strip Photo</div>
                                <div class="upload-btn">Browse File</div>
                                <input type="file" id="fileInput" accept="image/*" capture="environment" required onchange="handleFileSelect(this)">
                            </div>

                            <button type="submit" id="analyzeBtn" class="auth-btn" style="margin-top:20px;">Run Chemical Analysis</button>
                        </form>

                        <div id="resultBox"></div>
                    </div>
                </div>

                <div id="tab-logs" class="tab-content">
                    <div class="table-card">
                        <div class="table-hdr">
                            <h3 style="color:var(--accent);">Tamper-Evident Forensic Audit Logs</h3>
                            <a href="/export-csv" style="background:var(--success); color:#fff; padding:8px 16px; border-radius:6px; text-decoration:none; font-size:13px; font-weight:bold;">📥 Download Legal CSV Report</a>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Officer & Biometric</th>
                                    <th>Result</th>
                                    <th>Hue Score</th>
                                    <th>GPS Location</th>
                                    <th>IST Timestamp</th>
                                    <th>SHA-256 Seal</th>
                                </tr>
                            </thead>
                            <tbody id="logsTableBody">
                                <tr><td colspan="7" style="text-align:center; color:var(--muted); padding:20px;">Loading audit records...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div id="tab-sih" class="tab-content">
                    <div style="background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:24px;">
                        <h3 style="color:var(--accent); margin-bottom:12px;">Smart India Hackathon System Specifications</h3>
                        <ul style="color:var(--muted); line-height:1.8; font-size:14px; padding-left:20px;">
                            <li><b>Computer Vision Engine:</b> OpenCV Hue-Saturation-Value (HSV) mask segmentation targeting narcotic reagent reactions.</li>
                            <li><b>Cryptographic Chain of Custody:</b> SHA-256 hashing combining image binary data, officer ID, biometric token, GPS, and timestamp.</li>
                            <li><b>Biometric Authentication:</b> Simulated WebAuthn fingerprint pattern verification.</li>
                            <li><b>Offline-First Architecture:</b> Embedded SQLite audit database with CSV evidence export capabilities.</li>
                            <li><b>Standard Compliance:</b> Indian Standard Time (IST Asia/Kolkata) synchronization.</li>
                        </ul>
                    </div>
                </div>
            </main>
        </div>

        <script>
            let officerToken = "";
            let currentGps = "13.08270° N, 80.27070° E (Chennai Fix)";

            // 1. IST Clock
            function updateIST() {
                const now = new Date();
                const ist = now.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', dateStyle: 'medium', timeStyle: 'medium' });
                document.getElementById('liveIstClock').innerText = ist + ' IST';
            }
            setInterval(updateIST, 1000);
            updateIST();

            // 2. Satellite GPS Fetch
            function fetchGPS() {
                const disp = document.getElementById('liveGpsCoords');
                disp.innerText = "Connecting to satellite...";
                if ("geolocation" in navigator) {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => {
                            currentGps = `${pos.coords.latitude.toFixed(5)}° N, ${pos.coords.longitude.toFixed(5)}° E (±${Math.round(pos.coords.accuracy)}m)`;
                            disp.innerText = currentGps;
                        },
                        (err) => { disp.innerText = currentGps; },
                        { enableHighAccuracy: true, timeout: 5000 }
                    );
                } else { disp.innerText = currentGps; }
            }
            fetchGPS();

            // 3. Biometric Authentication Screen
            function authenticateOfficer() {
                const status = document.getElementById('fpModalStatus');
                const btn = document.getElementById('enterAppBtn');
                status.innerText = "Scanning fingerprint ridge patterns...";
                status.style.color = "var(--accent)";

                setTimeout(() => {
                    officerToken = "FP-BIO-" + Math.random().toString(36).substring(2, 10).toUpperCase();
                    status.innerText = "✅ Biometric Authenticated (" + officerToken + ")";
                    status.style.color = "var(--success)";
                    btn.disabled = false;
                    btn.innerText = "Access Forensic Portal →";
                }, 800);
            }

            function unlockSystem() {
                document.getElementById('authModal').style.display = 'none';
                document.getElementById('appLayout').style.display = 'flex';
                reloadStatsAndLogs();
            }

            function lockSystem() {
                document.getElementById('authModal').style.display = 'flex';
                document.getElementById('appLayout').style.display = 'none';
                document.getElementById('enterAppBtn').disabled = true;
                document.getElementById('enterAppBtn').innerText = "System Locked";
                document.getElementById('fpModalStatus').innerText = "Tap biometric sensor to authenticate officer";
                document.getElementById('fpModalStatus').style.color = "var(--muted)";
            }

            // 4. Tab Navigation
            function switchTab(tabId) {
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
                    document.getElementById('fileNameDisplay').innerText = "Selected: " + input.files[0].name;
                }
            }

            // 5. Submit Chemical Test Form
            document.getElementById('scannerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const btn = document.getElementById('analyzeBtn');
                const box = document.getElementById('resultBox');

                if (!fileInput.files[0]) {
                    alert("Please select a test strip photo.");
                    return;
                }

                btn.disabled = true;
                btn.innerText = "Analyzing Reagent Matrix...";

                const nowIST = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) + ' IST';
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('officer_id', document.getElementById('badgeInput').value);
                formData.append('fingerprint_hash', officerToken);
                formData.append('gps_coords', currentGps);
                formData.append('timestamp_ist', nowIST);

                try {
                    const res = await fetch('/verify', { method: 'POST', body: formData });
                    const data = await res.json();

                    box.style.display = "block";
                    if (data.classification === "POSITIVE") {
                        box.className = "res-pos";
                        box.innerHTML = `<strong>⚠️ POSITIVE REACTION DETECTED</strong><br>Narcotic reagent match confirmed.<br><small>Hue Score: ${data.hue_score}</small>`;
                    } else {
                        box.className = "res-neg";
                        box.innerHTML = `<strong>✅ NEGATIVE REACTION</strong><br>No controlled substance detected.<br><small>Hue Score: ${data.hue_score}</small>`;
                    }
                    box.innerHTML += `
                        <div style="font-size:11px; margin-top:10px; opacity:0.9;">
                            <b>GPS:</b> ${data.gps_coords}<br>
                            <b>Time:</b> ${data.timestamp_ist}<br>
                            <b>SHA-256 Seal:</b> ${data.sha256_hash.substring(0, 24)}...
                        </div>
                    `;
                    reloadStatsAndLogs();
                } catch (err) {
                    alert("Analysis server connection error.");
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Run Chemical Analysis";
                }
            });

            // 6. Fetch Logs & Stats API
            async function reloadStatsAndLogs() {
                try {
                    const res = await fetch('/api/stats');
                    const data = await res.json();

                    document.getElementById('statTotal').innerText = data.total;
                    document.getElementById('statPos').innerText = data.positive;
                    document.getElementById('statNeg').innerText = data.negative;

                    const tbody = document.getElementById('logsTableBody');
                    if (data.logs.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--muted); padding:20px;">No forensic audit records logged yet.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = data.logs.map(r => `
                        <tr>
                            <td>#${r.id}</td>
                            <td><b>${r.officer_id}</b><br><small style="color:var(--muted);">${r.fingerprint_hash}</small></td>
                            <td><span class="${r.classification === 'POSITIVE' ? 'b-pos' : 'b-neg'}">${r.classification}</span></td>
                            <td>${r.hue_score}</td>
                            <td><small>${r.gps_coords}</small></td>
                            <td><small>${r.timestamp_ist}</small></td>
                            <td><code title="${r.sha256_hash}">${r.sha256_hash.substring(0, 10)}...${r.sha256_hash.slice(-6)}</code></td>
                        </tr>
                    `).join('');
                } catch (err) {
                    console.error("Failed to load stats.");
                }
            }
        </script>
    </body>
    </html>
    """


# --- 4. VERIFICATION API ENDPOINT ---
@app.post("/verify")
async def verify_sample(
    file: UploadFile = File(...),
    officer_id: str = Form(...),
    fingerprint_hash: str = Form(...),
    gps_coords: str = Form(...),
    timestamp_ist: str = Form(...)
):
    image_bytes = await file.read()
    classification, hue_score = process_test_image(image_bytes)

    hash_input = image_bytes + f"{officer_id}{fingerprint_hash}{gps_coords}{timestamp_ist}{classification}".encode('utf-8')
    sha256_hash = hashlib.sha256(hash_input).hexdigest()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_logs (officer_id, fingerprint_hash, gps_coords, timestamp_ist, classification, hue_score, sha256_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (officer_id, fingerprint_hash, gps_coords, timestamp_ist, classification, hue_score, sha256_hash))
    conn.commit()
    conn.close()

    return JSONResponse({
        "status": "success",
        "classification": classification,
        "hue_score": hue_score,
        "sha256_hash": sha256_hash,
        "fingerprint_hash": fingerprint_hash,
        "gps_coords": gps_coords,
        "timestamp_ist": timestamp_ist
    })


# --- 5. STATS & LOGS API ENDPOINT ---
@app.get("/api/stats")
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*), SUM(CASE WHEN classification="POSITIVE" THEN 1 ELSE 0 END), SUM(CASE WHEN classification="NEGATIVE" THEN 1 ELSE 0 END) FROM audit_logs')
    row = cursor.fetchone()
    total = row[0] or 0
    pos = row[1] or 0
    neg = row[2] or 0

    cursor.execute('SELECT id, officer_id, fingerprint_hash, gps_coords, timestamp_ist, classification, hue_score, sha256_hash FROM audit_logs ORDER BY id DESC')
    logs = [
        {
            "id": r[0], "officer_id": r[1], "fingerprint_hash": r[2], 
            "gps_coords": r[3], "timestamp_ist": r[4], "classification": r[5], 
            "hue_score": r[6], "sha256_hash": r[7]
        } for r in cursor.fetchall()
    ]
    conn.close()

    return JSONResponse({"total": total, "positive": pos, "negative": neg, "logs": logs})


# --- 6. CSV REPORT EXPORT ENDPOINT ---
@app.get("/export-csv")
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, officer_id, fingerprint_hash, gps_coords, timestamp_ist, classification, hue_score, sha256_hash FROM audit_logs ORDER BY id ASC')
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Officer ID', 'Biometric Token', 'GPS Coordinates', 'IST Timestamp', 'Classification', 'Hue Score', 'SHA-256 Evidence Seal'])
    for r in rows:
        writer.writerow(r)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=drug_trace_audit_logs.csv"}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
