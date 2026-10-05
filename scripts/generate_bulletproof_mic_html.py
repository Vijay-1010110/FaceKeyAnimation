import json
import base64
import os
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

mp3_path = "test_results_epoch10/steve_jobs_clip.mp3"
with open(mp3_path, "rb") as f:
    mp3_b64 = base64.b64encode(f.read()).decode("ascii")

out_html = "C:/Users/Vijay/.gemini/antigravity/brain/8cda1bd6-80d5-49fe-bb76-e27120a22c87/live_mic_face_skeleton.html"

html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FaceKey Live Voice & Mic 3D Skeleton Reaction</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{
      background: #090d16;
      color: #e2e8f0;
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      margin: 0;
      overflow-x: hidden;
    }}
    .glow-cyan {{
      box-shadow: 0 0 30px rgba(6, 182, 212, 0.25);
    }}
    .slider {{
      accent-color: #06b6d4;
    }}
    .pulse-live {{
      animation: pulseLive 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}
    @keyframes pulseLive {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.5; transform: scale(1.1); }}
    }}
  </style>
</head>
<body class="p-4 flex flex-col items-center justify-center min-h-screen">
  <div class="w-full max-w-4xl bg-slate-900/90 border border-slate-800 rounded-2xl p-6 glow-cyan backdrop-blur-xl flex flex-col gap-6">
    
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-slate-800/80 pb-4">
      <div class="flex items-center gap-3">
        <div id="mic-status-dot" class="w-3.5 h-3.5 rounded-full bg-slate-600 transition-colors"></div>
        <div>
          <h1 class="text-lg font-bold text-white tracking-wide flex items-center gap-2">
            FaceKey Real-Time Voice Reaction Engine
            <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">Live Acoustic Kinematics</span>
          </h1>
          <p class="text-xs text-slate-400">Speak into your microphone or play the embedded test speech</p>
        </div>
      </div>
      <div class="text-right">
        <span id="mic-state-badge" class="text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 border border-slate-700">
          STANDBY
        </span>
      </div>
    </div>

    <!-- Main Viewport Area -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
      
      <!-- 3D Canvas Viewport (Span 2) -->
      <div class="md:col-span-2 relative bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex flex-col items-center justify-center aspect-square md:aspect-auto md:h-[480px]">
        <canvas id="face-canvas" width="600" height="480" class="cursor-grab active:cursor-grabbing"></canvas>
        
        <!-- Controls Hint -->
        <div class="absolute top-3 left-3 text-[11px] text-slate-400 bg-slate-900/80 px-2.5 py-1 rounded-md pointer-events-none border border-slate-700/60 shadow">
          🖱️ Drag mouse to rotate 3D head
        </div>

        <!-- Latency / Rate HUD -->
        <div class="absolute top-3 right-3 text-xs font-mono text-cyan-400 bg-slate-900/85 px-2.5 py-1 rounded-md border border-cyan-500/30 flex items-center gap-2">
          <span id="hud-fps">60 FPS</span> | <span id="hud-status">STANDBY</span>
        </div>

        <!-- Real-Time Audio Energy Wave Visualizer Bar -->
        <div class="absolute bottom-3 left-4 right-4 bg-slate-900/90 border border-slate-700/60 rounded-lg p-2.5 flex items-center gap-3">
          <span class="text-xs font-mono text-slate-400 font-semibold whitespace-nowrap">VOICE RMS:</span>
          <div class="flex-1 h-3.5 bg-slate-800 rounded-full overflow-hidden relative">
            <div id="audio-energy-bar" class="h-full bg-gradient-to-r from-cyan-500 via-emerald-400 to-amber-400 transition-all duration-75" style="width: 0%;"></div>
          </div>
          <span id="audio-energy-text" class="text-xs font-mono text-amber-300 font-bold w-12 text-right">0.000</span>
        </div>
      </div>

      <!-- Right Telemetry & Articulation Controls Panel -->
      <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 h-full">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800 pb-2 flex justify-between">
          <span>Live Blendshapes</span>
          <span id="live-indicator" class="text-slate-500 font-normal">Waiting</span>
        </h2>

        <!-- Slider: Mic Gain / Sensitivity -->
        <div class="bg-slate-900/60 border border-slate-800 p-2.5 rounded-lg flex flex-col gap-1.5">
          <div class="flex justify-between text-xs">
            <span class="text-slate-300 font-semibold">Microphone Sensitivity</span>
            <span id="val-sens" class="font-mono text-cyan-300 font-bold">2.5x</span>
          </div>
          <input type="range" id="sens-slider" min="0.5" max="6.0" step="0.1" value="2.5" class="slider cursor-pointer w-full">
          <span class="text-[10px] text-slate-500">Boosts quiet speech or whisper volume</span>
        </div>

        <!-- Slider: Articulation Scale -->
        <div class="bg-slate-900/60 border border-slate-800 p-2.5 rounded-lg flex flex-col gap-1.5">
          <div class="flex justify-between text-xs">
            <span class="text-slate-300 font-semibold">Mouth Articulation Boost</span>
            <span id="val-scale" class="font-mono text-amber-300 font-bold">1.5x</span>
          </div>
          <input type="range" id="scale-slider" min="0.5" max="3.0" step="0.1" value="1.5" class="slider cursor-pointer w-full">
          <span class="text-[10px] text-slate-500">Controls how wide the mouth and teeth part</span>
        </div>

        <!-- Meter: jawOpen / Articulation -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-amber-400 font-semibold">jawOpen (Mouth Opening)</span>
            <span id="val-jaw" class="font-mono text-amber-300 font-bold">0.000</span>
          </div>
          <div class="h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-jaw" class="h-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Meter: Speech Drive Envelope -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-cyan-400 font-semibold">Syllabic Energy Envelope</span>
            <span id="val-drive" class="font-mono text-cyan-300 font-bold">0.000</span>
          </div>
          <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-drive" class="h-full bg-cyan-500 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Diagnostics Log Box -->
        <div class="mt-auto bg-slate-900/80 border border-slate-800 rounded-lg p-2.5 text-[11px] font-mono flex flex-col gap-1">
          <div class="flex justify-between text-slate-400">
            <span>Audio Context:</span>
            <span id="diag-ctx" class="text-slate-300">uninitialized</span>
          </div>
          <div class="flex justify-between text-slate-400">
            <span>Input Signal:</span>
            <span id="diag-signal" class="text-slate-300">0.000</span>
          </div>
          <div id="diag-msg" class="text-cyan-400 text-[10px] truncate">Ready to connect</div>
        </div>
      </div>
    </div>

    <!-- Live Microphone Control Bar -->
    <div class="bg-slate-950/90 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
      
      <div class="flex items-center gap-3 flex-wrap">
        <button id="btn-toggle-mic" class="px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all flex items-center gap-2 shadow-lg shadow-cyan-500/20 active:scale-95 cursor-pointer">
          <span id="btn-mic-icon">🎙️</span> <span id="btn-mic-label">Start Live Microphone</span>
        </button>

        <button id="btn-test-speech" class="px-5 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-amber-300 text-xs font-semibold transition-colors flex items-center gap-2 cursor-pointer border border-slate-700">
          <span id="btn-test-icon">🔊</span> <span id="btn-test-label">Play Test Audio (Steve Jobs Speech)</span>
        </button>

        <button id="btn-reset-view" class="px-4 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors">
          🔄 Reset 3D View
        </button>
      </div>

      <div class="text-xs text-slate-400 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span>Runs 100% locally in browser Web Audio • High-precision 60 FPS visualizer</span>
      </div>

    </div>

  </div>

  <script>
    const canvas = document.getElementById('face-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const sensSlider = document.getElementById('sens-slider');
    const scaleSlider = document.getElementById('scale-slider');
    const btnToggleMic = document.getElementById('btn-toggle-mic');
    const btnTestSpeech = document.getElementById('btn-test-speech');
    const btnResetView = document.getElementById('btn-reset-view');

    const diagCtx = document.getElementById('diag-ctx');
    const diagSignal = document.getElementById('diag-signal');
    const diagMsg = document.getElementById('diag-msg');

    let isMicActive = false;
    let isTestPlaying = false;
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let dataArray = null;

    let testAudioBuffer = null;
    let testBufferSource = null;

    let rotX = 0.0;
    let rotY = 0.0;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    // Base64 MP3 Audio Data
    const B64_AUDIO = "{mp3_b64}";

    function base64ToArrayBuffer(base64) {{
      const binary_string = window.atob(base64);
      const len = binary_string.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{
        bytes[i] = binary_string.charCodeAt(i);
      }}
      return bytes.buffer;
    }}

    // Mouse 3D Orbit
    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {{
      if (!isDragging) return;
      rotY += (e.clientX - lastMouseX) * 0.008;
      rotX += (e.clientY - lastMouseY) * 0.008;
      rotX = Math.max(-0.6, Math.min(0.6, rotX));
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});

    btnResetView.addEventListener('click', () => {{
      rotX = 0.0;
      rotY = 0.0;
    }});

    sensSlider.addEventListener('input', (e) => {{
      document.getElementById('val-sens').innerText = e.target.value + 'x';
    }});
    scaleSlider.addEventListener('input', (e) => {{
      document.getElementById('val-scale').innerText = e.target.value + 'x';
    }});

    // 3D Canonical Skeleton Geometry
    const NEUTRAL_CONTOUR = [
      [0.0, -0.82, -0.1], [-0.55, -0.52, -0.22], [-0.62, -0.05, -0.18], [-0.52, 0.40, -0.14],
      [0.0, 0.76, -0.05], [0.52, 0.40, -0.14], [0.62, -0.05, -0.18], [0.55, -0.52, -0.22]
    ];
    const NEUTRAL_L_BROW = [[-0.45, -0.42, -0.05], [-0.34, -0.48, -0.02], [-0.22, -0.47, 0.0], [-0.10, -0.42, 0.02]];
    const NEUTRAL_R_BROW = [[0.10, -0.42, 0.02], [0.22, -0.47, 0.0], [0.34, -0.48, -0.02], [0.45, -0.42, -0.05]];
    const NEUTRAL_L_EYE = [[-0.40, -0.28, -0.05], [-0.31, -0.34, -0.02], [-0.19, -0.34, 0.0], [-0.13, -0.28, 0.02], [-0.19, -0.22, 0.0], [-0.31, -0.22, -0.02]];
    const NEUTRAL_R_EYE = [[0.13, -0.28, 0.02], [0.19, -0.34, 0.0], [0.31, -0.34, -0.02], [0.40, -0.28, -0.05], [0.31, -0.22, -0.02], [0.19, -0.22, 0.0]];
    const NEUTRAL_NOSE = [[0.0, -0.35, 0.08], [0.0, -0.12, 0.15], [0.0, 0.06, 0.22]];
    const NEUTRAL_NOSTRILS = [[-0.12, 0.12, 0.14], [0.0, 0.08, 0.20], [0.12, 0.12, 0.14]];
    const NEUTRAL_LIPS_OUTER = [
      [-0.26, 0.35, 0.05], [-0.16, 0.28, 0.10], [0.0, 0.26, 0.12], [0.16, 0.28, 0.10],
      [0.26, 0.35, 0.05], [0.16, 0.42, 0.08], [0.0, 0.44, 0.10], [-0.16, 0.42, 0.08]
    ];
    const NEUTRAL_LIPS_INNER = [
      [-0.20, 0.35, 0.06], [0.0, 0.31, 0.10], [0.20, 0.35, 0.06], [0.0, 0.38, 0.08]
    ];

    function project(pt3d, extraPitch = 0.0) {{
      let [x, y, z] = pt3d;
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      const totalRotX = rotX + extraPitch;
      const cosX = Math.cos(totalRotX), sinX = Math.sin(totalRotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const fov = 3.2;
      const scale = fov / (fov + z2);
      const sx = (x1 * scale) * 210 + width / 2;
      const sy = (y2 * scale) * 210 + height / 2 + 10;
      return [sx, sy];
    }}

    function drawWire(pts, color, lineWidth = 1.5, close = false, extraPitch = 0.0) {{
      if (pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      const p0 = project(pts[0], extraPitch);
      ctx.moveTo(p0[0], p0[1]);
      for (let i = 1; i < pts.length; i++) {{
        const p = project(pts[i], extraPitch);
        ctx.lineTo(p[0], p[1]);
      }}
      if (close) ctx.closePath();
      ctx.stroke();

      for (let i = 0; i < pts.length; i++) {{
        const p = project(pts[i], extraPitch);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p[0], p[1], lineWidth * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    // Dynamic Kinematics State
    let smoothDrive = 0.0;
    let peakEnergy = 0.025;

    function renderFace(energyRms) {{
      ctx.clearRect(0, 0, width, height);

      // Grid
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 35) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }}
      for (let y = 0; y < height; y += 35) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }}

      const sens = parseFloat(sensSlider.value);
      const boost = parseFloat(scaleSlider.value);

      // Adaptive speech energy envelope
      const adjustedEnergy = energyRms * sens;
      diagSignal.innerText = adjustedEnergy.toFixed(4);

      if (adjustedEnergy > peakEnergy) {{
        peakEnergy = 0.90 * peakEnergy + 0.10 * adjustedEnergy;
      }} else {{
        peakEnergy = Math.max(0.012, 0.998 * peakEnergy);
      }}

      const normEnergy = Math.min(1.0, adjustedEnergy / Math.max(1e-4, peakEnergy * 0.85));
      const rawDrive = Math.max(0.0, normEnergy);

      // Attack / Release envelope
      const alpha = rawDrive > smoothDrive ? 0.70 : 0.22;
      smoothDrive = alpha * rawDrive + (1.0 - alpha) * smoothDrive;

      // Kinematic deformations
      const jaw = smoothDrive * 0.45 * boost;
      const speechNod = smoothDrive * 0.06;
      const upperLift = jaw * 0.18;
      const browUp = smoothDrive * 0.12 * boost;

      // Deform Contour
      const contour = NEUTRAL_CONTOUR.map((p, idx) => {{
        let [x, y, z] = p;
        if (idx === 4) {{ y += jaw * 1.35; z -= jaw * 0.15; }}
        if (idx === 3 || idx === 5) {{ y += jaw * 0.55; }}
        return [x, y, z];
      }});

      const l_brow = NEUTRAL_L_BROW.map(([x, y, z]) => [x, y - browUp, z]);
      const r_brow = NEUTRAL_R_BROW.map(([x, y, z]) => [x, y - browUp, z]);
      const l_eye = NEUTRAL_L_EYE.map(([x, y, z]) => [x, y, z]);
      const r_eye = NEUTRAL_R_EYE.map(([x, y, z]) => [x, y, z]);

      const lips_outer = NEUTRAL_LIPS_OUTER.map(([x, y, z], idx) => {{
        if (idx >= 5) y += jaw * 0.90;
        if (idx >= 1 && idx <= 3) y -= upperLift;
        return [x, y, z];
      }});

      const lips_inner = NEUTRAL_LIPS_INNER.map(([x, y, z], idx) => {{
        if (idx === 3) y += jaw * 0.85;
        if (idx === 1) y -= upperLift;
        return [x, y, z];
      }});

      // Draw skeleton
      drawWire(contour, '#38bdf8', 2, true, speechNod);
      drawWire(l_brow, '#60a5fa', 2.2, false, speechNod);
      drawWire(r_brow, '#60a5fa', 2.2, false, speechNod);
      drawWire(l_eye, '#38bdf8', 2, true, speechNod);
      drawWire(r_eye, '#38bdf8', 2, true, speechNod);
      drawWire(NEUTRAL_NOSE, '#38bdf8', 1.5, false, speechNod);
      drawWire(NEUTRAL_NOSTRILS, '#38bdf8', 1.5, false, speechNod);

      // Pupils
      const lp = project([-0.29, -0.28, 0.0], speechNod);
      const rp = project([0.27, -0.28, 0.0], speechNod);
      ctx.fillStyle = '#67e8f9';
      ctx.beginPath(); ctx.arc(lp[0], lp[1], 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(rp[0], rp[1], 3.5, 0, Math.PI * 2); ctx.fill();

      // Teeth & Mouth Cavity
      const topLipP = project(lips_inner[1], speechNod);
      const botLipP = project(lips_inner[3], speechNod);
      const mouthGapPx = botLipP[1] - topLipP[1];

      if (mouthGapPx > 10) {{
        // Inner cavity
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        const p0 = project(lips_inner[0], speechNod);
        ctx.moveTo(p0[0], p0[1]);
        for (let i = 1; i < lips_inner.length; i++) {{
          const pi = project(lips_inner[i], speechNod);
          ctx.lineTo(pi[0], pi[1]);
        }}
        ctx.closePath();
        ctx.fill();

        // Upper teeth
        const teethW = Math.min(44, 30 * boost);
        ctx.fillStyle = '#f8fafc';
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(topLipP[0] - teethW, topLipP[1] + 2, teethW * 2, 7);
        ctx.fill();
        ctx.stroke();

        for (let tx = -teethW + 7; tx < teethW; tx += 8) {{
          ctx.beginPath();
          ctx.moveTo(topLipP[0] + tx, topLipP[1] + 2);
          ctx.lineTo(topLipP[0] + tx, topLipP[1] + 9);
          ctx.stroke();
        }}

        // Lower teeth
        if (mouthGapPx > 18) {{
          ctx.fillStyle = '#e2e8f0';
          ctx.beginPath();
          ctx.rect(botLipP[0] - teethW, botLipP[1] - 8, teethW * 2, 6);
          ctx.fill();
          ctx.stroke();
          for (let tx = -teethW + 7; tx < teethW; tx += 8) {{
            ctx.beginPath();
            ctx.moveTo(botLipP[0] + tx, botLipP[1] - 8);
            ctx.lineTo(botLipP[0] + tx, botLipP[1] - 2);
            ctx.stroke();
          }}
        }}
      }}

      // Lips
      drawWire(lips_outer, '#f43f5e', 2.8, true, speechNod);
      drawWire(lips_inner, '#fb7185', 1.8, true, speechNod);

      // Update Telemetry
      document.getElementById('val-jaw').innerText = jaw.toFixed(3);
      document.getElementById('bar-jaw').style.width = Math.min(100, jaw * 220) + '%';

      document.getElementById('val-drive').innerText = smoothDrive.toFixed(3);
      document.getElementById('bar-drive').style.width = Math.min(100, smoothDrive * 100) + '%';

      document.getElementById('audio-energy-text').innerText = adjustedEnergy.toFixed(3);
      document.getElementById('audio-energy-bar').style.width = Math.min(100, adjustedEnergy * 450) + '%';

      const isSpeaking = smoothDrive > 0.15;
      const badge = document.getElementById('mic-state-badge');
      if (isSpeaking) {{
        badge.innerText = '🗣️ VOICE ACTIVE (TALKING)';
        badge.className = 'text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      }} else if (isMicActive || isTestPlaying) {{
        badge.innerText = '👂 LISTENING (SILENCE)';
        badge.className = 'text-xs font-semibold px-2.5 py-1 rounded-md bg-cyan-500/20 text-cyan-300 border border-cyan-500/30';
      }} else {{
        badge.innerText = 'STANDBY';
        badge.className = 'text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 border border-slate-700';
      }}
    }}

    // Audio Analysis Loop
    let animFrameId = null;

    function audioLoop() {{
      if (!isMicActive && !isTestPlaying) return;

      analyser.getFloatTimeDomainData(dataArray);

      // Compute RMS Energy
      let sumSq = 0;
      for (let i = 0; i < dataArray.length; i++) {{
        sumSq += dataArray[i] * dataArray[i];
      }}
      const rms = Math.sqrt(sumSq / dataArray.length);

      renderFace(rms);
      animFrameId = requestAnimationFrame(audioLoop);
    }}

    async function initAudioContext() {{
      if (!audioCtx) {{
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }}
      if (audioCtx.state === 'suspended') {{
        await audioCtx.resume();
      }}
      diagCtx.innerText = audioCtx.state;
    }}

    async function startMicrophone() {{
      if (isTestPlaying) stopTestAudio();

      diagMsg.innerText = 'Requesting mic permission...';
      try {{
        await initAudioContext();

        // Standard unconstrained audio request for 100% browser compatibility
        micStream = await navigator.mediaDevices.getUserMedia({{
          audio: {{
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: true
          }},
          video: false
        }});

        const source = audioCtx.createMediaStreamSource(micStream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        dataArray = new Float32Array(analyser.fftSize);
        source.connect(analyser);

        isMicActive = true;
        document.getElementById('btn-mic-icon').innerText = '⏹️';
        document.getElementById('btn-mic-label').innerText = 'Stop Microphone';
        btnToggleMic.classList.replace('bg-cyan-500', 'bg-rose-500');
        btnToggleMic.classList.replace('hover:bg-cyan-400', 'hover:bg-rose-400');
        btnToggleMic.classList.replace('text-slate-950', 'text-white');
        document.getElementById('mic-status-dot').className = 'w-3.5 h-3.5 rounded-full bg-emerald-400 pulse-live';
        document.getElementById('hud-status').innerText = 'LIVE MIC';
        document.getElementById('live-indicator').innerText = 'Connected';
        document.getElementById('live-indicator').className = 'text-emerald-400 font-semibold';
        diagMsg.innerText = 'Microphone streaming live!';

        audioLoop();
      }} catch (err) {{
        console.error('Microphone error:', err);
        diagMsg.innerText = 'Error: ' + err.message;
        alert('Could not open microphone: ' + err.message + '\\n\\nTip: If you opened this file directly, please use http://localhost:8765/live_mic_face_skeleton.html to allow browser permissions.');
      }}
    }}

    function stopMicrophone() {{
      isMicActive = false;
      if (animFrameId) cancelAnimationFrame(animFrameId);
      if (micStream) {{
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
      }}

      document.getElementById('btn-mic-icon').innerText = '🎙️';
      document.getElementById('btn-mic-label').innerText = 'Start Live Microphone';
      btnToggleMic.classList.replace('bg-rose-500', 'bg-cyan-500');
      btnToggleMic.classList.replace('hover:bg-rose-400', 'hover:bg-cyan-400');
      btnToggleMic.classList.replace('text-white', 'text-slate-950');
      document.getElementById('mic-status-dot').className = 'w-3.5 h-3.5 rounded-full bg-slate-600';
      document.getElementById('hud-status').innerText = 'STANDBY';
      document.getElementById('live-indicator').innerText = 'Waiting';
      document.getElementById('live-indicator').className = 'text-slate-500 font-normal';
      diagMsg.innerText = 'Microphone stopped';

      renderFace(0.0);
    }}

    // Bulletproof In-Memory Web Audio Test Player
    async function startTestAudio() {{
      if (isMicActive) stopMicrophone();

      diagMsg.innerText = 'Loading in-memory speech audio...';
      try {{
        await initAudioContext();

        if (!testAudioBuffer) {{
          const arrayBuf = base64ToArrayBuffer(B64_AUDIO);
          testAudioBuffer = await audioCtx.decodeAudioData(arrayBuf);
        }}

        testBufferSource = audioCtx.createBufferSource();
        testBufferSource.buffer = testAudioBuffer;
        testBufferSource.loop = true;

        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        dataArray = new Float32Array(analyser.fftSize);

        testBufferSource.connect(analyser);
        analyser.connect(audioCtx.destination);
        testBufferSource.start(0);

        isTestPlaying = true;
        document.getElementById('btn-test-icon').innerText = '⏹️';
        document.getElementById('btn-test-label').innerText = 'Stop Test Audio';
        btnTestSpeech.classList.replace('bg-slate-800', 'bg-amber-500');
        btnTestSpeech.classList.replace('text-amber-300', 'text-slate-950');
        document.getElementById('mic-status-dot').className = 'w-3.5 h-3.5 rounded-full bg-amber-400 pulse-live';
        document.getElementById('hud-status').innerText = 'TEST SPEECH';
        document.getElementById('live-indicator').innerText = 'Playing';
        document.getElementById('live-indicator').className = 'text-amber-400 font-semibold';
        diagMsg.innerText = 'Playing in-memory speech track';

        audioLoop();
      }} catch (err) {{
        console.error('Test audio error:', err);
        diagMsg.innerText = 'Test Error: ' + err.message;
        alert('Test audio error: ' + err.message);
      }}
    }}

    function stopTestAudio() {{
      isTestPlaying = false;
      if (animFrameId) cancelAnimationFrame(animFrameId);
      if (testBufferSource) {{
        try {{ testBufferSource.stop(); }} catch(e) {{}}
        testBufferSource = null;
      }}

      document.getElementById('btn-test-icon').innerText = '🔊';
      document.getElementById('btn-test-label').innerText = 'Play Test Audio (Steve Jobs Speech)';
      btnTestSpeech.classList.replace('bg-amber-500', 'bg-slate-800');
      btnTestSpeech.classList.replace('text-slate-950', 'text-amber-300');
      document.getElementById('mic-status-dot').className = 'w-3.5 h-3.5 rounded-full bg-slate-600';
      document.getElementById('hud-status').innerText = 'STANDBY';
      document.getElementById('live-indicator').innerText = 'Waiting';
      document.getElementById('live-indicator').className = 'text-slate-500 font-normal';
      diagMsg.innerText = 'Test audio stopped';

      renderFace(0.0);
    }}

    btnToggleMic.addEventListener('click', () => {{
      if (isMicActive) stopMicrophone(); else startMicrophone();
    }});

    btnTestSpeech.addEventListener('click', () => {{
      if (isTestPlaying) stopTestAudio(); else startTestAudio();
    }});

    // Initial neutral frame
    renderFace(0.0);
  </script>
</body>
</html>
"""

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_code)

print(f"[OK] Live mic player written to {out_html}")
