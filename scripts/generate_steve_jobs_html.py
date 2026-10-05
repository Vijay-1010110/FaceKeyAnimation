import json
import base64
import os

telemetry_path = "test_results_epoch10/steve_jobs_face_skeleton_reaction_telemetry.json"
mp3_path = "test_results_epoch10/steve_jobs_clip.mp3"
out_html = "C:/Users/Vijay/.gemini/antigravity/brain/8cda1bd6-80d5-49fe-bb76-e27120a22c87/face_skeleton_player_steve_jobs.html"

with open(telemetry_path, "r", encoding="utf-8") as f:
    frames = json.load(f)

with open(mp3_path, "rb") as f:
    mp3_b64 = base64.b64encode(f.read()).decode("ascii")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FaceKey 3D Skeleton Reaction: Steve Jobs Speech</title>
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
      box-shadow: 0 0 25px rgba(6, 182, 212, 0.25);
    }}
    .slider {{
      accent-color: #06b6d4;
    }}
  </style>
</head>
<body class="p-4 flex flex-col items-center justify-center min-h-screen">
  <div class="w-full max-w-4xl bg-slate-900/90 border border-slate-800 rounded-2xl p-6 glow-cyan backdrop-blur-xl flex flex-col gap-6">
    
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-slate-800/80 pb-4">
      <div class="flex items-center gap-3">
        <div class="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></div>
        <div>
          <h1 class="text-lg font-bold text-white tracking-wide flex items-center gap-2">
            FaceKey 3D Skeleton Reaction: Steve Jobs Speech
            <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">Model Epoch 10</span>
          </h1>
          <p class="text-xs text-slate-400">“Today I want to tell you three stories from my life. That's it...”</p>
        </div>
      </div>
      <div class="text-right">
        <span id="status-badge" class="text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          REAL SPEECH SYNC
        </span>
      </div>
    </div>

    <!-- Main Viewport Area -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
      
      <!-- 3D Canvas Viewport (Span 2) -->
      <div class="md:col-span-2 relative bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex flex-col items-center justify-center aspect-square md:aspect-auto md:h-[460px]">
        <canvas id="face-canvas" width="600" height="460" class="cursor-grab active:cursor-grabbing"></canvas>
        
        <!-- Controls Hint -->
        <div class="absolute top-3 left-3 text-[11px] text-slate-500 bg-slate-900/70 px-2 py-1 rounded pointer-events-none border border-slate-800">
          Drag mouse to rotate in 3D
        </div>

        <!-- Frame Counter HUD -->
        <div class="absolute top-3 right-3 text-xs font-mono text-cyan-400 bg-slate-900/80 px-2.5 py-1 rounded border border-cyan-500/20">
          <span id="hud-time">0.00s</span> | Frame <span id="hud-frame">0</span>/{len(frames)}
        </div>

        <!-- Real-Time Audio Energy Wave Visualizer Bar -->
        <div class="absolute bottom-3 left-4 right-4 bg-slate-900/90 border border-slate-700/60 rounded-lg p-2.5 flex items-center gap-3">
          <span class="text-xs font-mono text-slate-400 font-semibold whitespace-nowrap">VOICE RMS:</span>
          <div class="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden relative">
            <div id="audio-energy-bar" class="h-full bg-gradient-to-r from-cyan-500 via-emerald-400 to-amber-400 transition-all duration-75" style="width: 15%;"></div>
          </div>
          <span id="audio-energy-text" class="text-xs font-mono text-amber-300 font-bold w-12 text-right">0.024</span>
        </div>
      </div>

      <!-- Right Telemetry & Blendshape Meters Panel -->
      <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 h-full">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 border-b border-slate-800 pb-2">
          Live ARKit Telemetry
        </h2>

        <!-- Meter: jawOpen -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-amber-400 font-semibold">jawOpen (Phonemes)</span>
            <span id="val-jaw" class="font-mono text-amber-300 font-bold">0.000</span>
          </div>
          <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-jaw" class="h-full bg-amber-500 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Meter: mouthPucker -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-purple-400 font-semibold">mouthPucker (Co-articulation)</span>
            <span id="val-pucker" class="font-mono text-purple-300 font-bold">0.000</span>
          </div>
          <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-pucker" class="h-full bg-purple-500 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Meter: mouthSmile -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-emerald-400 font-semibold">mouthSmile</span>
            <span id="val-smile" class="font-mono text-emerald-300 font-bold">0.000</span>
          </div>
          <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-smile" class="h-full bg-emerald-500 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Meter: browInnerUp -->
        <div class="flex flex-col gap-1">
          <div class="flex justify-between text-xs">
            <span class="text-cyan-400 font-semibold">browInnerUp (Inflection)</span>
            <span id="val-brow" class="font-mono text-cyan-300 font-bold">0.000</span>
          </div>
          <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div id="bar-brow" class="h-full bg-cyan-500 transition-all duration-75" style="width: 0%;"></div>
          </div>
        </div>

        <div class="mt-auto pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 leading-relaxed">
          <p><span class="text-slate-400 font-semibold">Neural Inference:</span> Speech acoustics fed directly to dual-layer LSTM network, predicting 52 ARKit blendshapes in real time.</p>
        </div>
      </div>
    </div>

    <!-- Playback Transport Bar -->
    <div class="bg-slate-950/90 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row items-center gap-4">
      
      <!-- Play / Pause & Loop Button -->
      <div class="flex items-center gap-2">
        <button id="btn-play" class="px-5 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-colors flex items-center gap-1.5">
          <span>▶</span> Play Speech Sync
        </button>
        <button id="btn-loop" class="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors">
          🔁 Loop: ON
        </button>
      </div>

      <!-- Scrubber -->
      <div class="flex-1 flex items-center gap-3 w-full">
        <span class="text-xs font-mono text-slate-400">0.0s</span>
        <input type="range" id="timeline-scrub" min="0" max="{len(frames) - 1}" value="0" class="slider flex-1 cursor-pointer">
        <span class="text-xs font-mono text-slate-400">{len(frames)/30:.1f}s</span>
      </div>

      <audio id="speech-audio" src="data:audio/mp3;base64,{mp3_b64}" preload="auto"></audio>
    </div>

  </div>

  <script>
    const frames = {json.dumps(frames)};
    const audioEl = document.getElementById('speech-audio');
    const canvas = document.getElementById('face-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    let rotX = 0.0;
    let rotY = 0.0;
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

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

    function project(pt3d) {{
      let [x, y, z] = pt3d;
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const fov = 3.2;
      const scale = fov / (fov + z2);
      const sx = (x1 * scale) * 210 + width / 2;
      const sy = (y2 * scale) * 210 + height / 2 + 10;
      return [sx, sy];
    }}

    function drawWire(pts, color, lineWidth = 1.5, close = false) {{
      if (pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      const p0 = project(pts[0]);
      ctx.moveTo(p0[0], p0[1]);
      for (let i = 1; i < pts.length; i++) {{
        const p = project(pts[i]);
        ctx.lineTo(p[0], p[1]);
      }}
      if (close) ctx.closePath();
      ctx.stroke();
    }}

    function render(fIdx) {{
      const fr = frames[fIdx];
      if (!fr) return;
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

      const jaw = fr.jaw * 0.7;
      const smile = (fr.smile_l + fr.smile_r) * 0.1;
      const pucker = fr.pucker * 0.15;
      const browUp = fr.brow_up * 0.12;

      const contour = NEUTRAL_CONTOUR.map((p, idx) => {{
        let [x, y, z] = p;
        if (idx === 4) y += jaw * 1.3;
        if (idx === 3 || idx === 5) y += jaw * 0.6;
        return [x, y, z];
      }});
      const l_brow = NEUTRAL_L_BROW.map(([x, y, z]) => [x, y - browUp, z]);
      const r_brow = NEUTRAL_R_BROW.map(([x, y, z]) => [x, y - browUp, z]);
      const l_eye = NEUTRAL_L_EYE.map(([x, y, z]) => [x, -0.28 + (y - (-0.28)) * (1.0 - fr.blink_l * 0.85), z]);
      const r_eye = NEUTRAL_R_EYE.map(([x, y, z]) => [x, -0.28 + (y - (-0.28)) * (1.0 - fr.blink_r * 0.85), z]);

      const lips_outer = NEUTRAL_LIPS_OUTER.map(([x, y, z], idx) => {{
        if (idx === 0) {{ x -= smile - pucker; y -= smile * 0.6; }}
        if (idx === 4) {{ x += smile - pucker; y -= smile * 0.6; }}
        if (idx >= 5) y += jaw;
        if (idx >= 1 && idx <= 3) y += jaw * 0.15;
        return [x, y, z];
      }});
      const lips_inner = NEUTRAL_LIPS_INNER.map(([x, y, z], idx) => {{
        if (idx === 0) x -= smile - pucker;
        if (idx === 2) x += smile - pucker;
        if (idx === 3) y += jaw * 0.9;
        return [x, y, z];
      }});

      drawWire(contour, '#38bdf8', 2, true);
      drawWire(l_brow, '#60a5fa', 2);
      drawWire(r_brow, '#60a5fa', 2);
      drawWire(l_eye, '#38bdf8', 2, true);
      drawWire(r_eye, '#38bdf8', 2, true);
      drawWire(NEUTRAL_NOSE, '#38bdf8', 1.5);
      drawWire(NEUTRAL_NOSTRILS, '#38bdf8', 1.5);
      drawWire(lips_outer, '#f43f5e', 2.5, true);
      drawWire(lips_inner, '#fb7185', 1.5, true);

      // Pupils
      const lp = project([-0.29, -0.28, 0.0]);
      const rp = project([0.27, -0.28, 0.0]);
      ctx.fillStyle = '#67e8f9';
      ctx.beginPath(); ctx.arc(lp[0], lp[1], 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(rp[0], rp[1], 3.5, 0, Math.PI * 2); ctx.fill();

      // Update Telemetry
      document.getElementById('hud-time').innerText = fr.t.toFixed(2) + 's';
      document.getElementById('hud-frame').innerText = fIdx;
      document.getElementById('timeline-scrub').value = fIdx;

      document.getElementById('val-jaw').innerText = fr.jaw.toFixed(3);
      document.getElementById('bar-jaw').style.width = Math.min(100, fr.jaw * 500) + '%';

      document.getElementById('val-pucker').innerText = fr.pucker.toFixed(3);
      document.getElementById('bar-pucker').style.width = Math.min(100, fr.pucker * 150) + '%';

      document.getElementById('val-smile').innerText = fr.smile_l.toFixed(3);
      document.getElementById('bar-smile').style.width = Math.min(100, fr.smile_l * 200) + '%';

      document.getElementById('val-brow').innerText = fr.brow_up.toFixed(3);
      document.getElementById('bar-brow').style.width = Math.min(100, fr.brow_up * 300) + '%';

      document.getElementById('audio-energy-text').innerText = fr.energy.toFixed(3);
      document.getElementById('audio-energy-bar').style.width = Math.min(100, fr.energy * 500) + '%';
    }}

    let isPlaying = false;
    let isLooping = true;
    let animTimer = null;

    function syncStep() {{
      if (!isPlaying) return;
      const curTime = audioEl.currentTime;
      let frameIdx = Math.floor(curTime * 30);
      if (frameIdx >= frames.length) {{
        if (isLooping) {{
          audioEl.currentTime = 0;
          audioEl.play();
          frameIdx = 0;
        }} else {{
          pause();
          return;
        }}
      }}
      render(frameIdx);
      animTimer = requestAnimationFrame(syncStep);
    }}

    function play() {{
      isPlaying = true;
      audioEl.play().catch(e => console.log('Audio autoplay prevented:', e));
      document.getElementById('btn-play').innerHTML = '<span>⏸</span> Pause';
      document.getElementById('btn-play').classList.replace('bg-cyan-500', 'bg-amber-500');
      document.getElementById('btn-play').classList.replace('hover:bg-cyan-400', 'hover:bg-amber-400');
      animTimer = requestAnimationFrame(syncStep);
    }}

    function pause() {{
      isPlaying = false;
      audioEl.pause();
      document.getElementById('btn-play').innerHTML = '<span>▶</span> Play Speech Sync';
      document.getElementById('btn-play').classList.replace('bg-amber-500', 'bg-cyan-500');
      document.getElementById('btn-play').classList.replace('hover:bg-amber-400', 'hover:bg-cyan-400');
      if (animTimer) cancelAnimationFrame(animTimer);
    }}

    document.getElementById('btn-play').addEventListener('click', () => {{
      if (isPlaying) pause(); else play();
    }});

    document.getElementById('btn-loop').addEventListener('click', () => {{
      isLooping = !isLooping;
      document.getElementById('btn-loop').innerText = '🔁 Loop: ' + (isLooping ? 'ON' : 'OFF');
      document.getElementById('btn-loop').classList.toggle('text-cyan-400', isLooping);
    }});

    document.getElementById('timeline-scrub').addEventListener('input', (e) => {{
      const idx = parseInt(e.target.value);
      audioEl.currentTime = idx / 30;
      render(idx);
    }});

    render(0);
  </script>
</body>
</html>
"""

os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[✓] HTML player written successfully to {out_html}")
