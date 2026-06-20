/* =====================================================
   CATSIUM — shared shell (looping video/gif background)
   Nav is server-rendered in base.html (so Flask can drive
   the active state, market pill, and route URLs).

   Round-7: replaced the gradient rainbow + CSS snow with a
   full-screen looping media layer. Drop a file in static/ named
   `background.mp4` (preferred) or `background.gif`. mp4 is tried
   first; on load error it falls back to the gif; if neither is
   present the dark html background shows. The .bgvid CSS pins it
   behind all content (z-index:-2) with a dim overlay for contrast.

   Round-8 (Onimai): added sakura-sparkle particle canvas,
   cursor sparkle trail, and gentle floating ambient emoji.

   Round-9 (Onimai refresh): the round-8 canvas had drifted into
   "rainbow snow" territory — multiple shapes (stars/crosses/petals),
   a wide color palette, and a glowy blur, plus a layer of floating
   emoji on top. Per feedback that's too busy, so this pass:
     - removes the floating emoji layer entirely (initFloatingDecos
       and its call are gone)
     - replaces the multi-shape "snow" with a single, calmer sakura
       petal shape, pastel-only palette, fewer particles, slower/
       gentler fall, and no heavy glow
     - adds a second, much sparser layer of quiet twinkling stars
       (fixed-ish position, slow fade in/out) since a still night
       sky reads as ambiance rather than clutter
     - softens the cursor sparkle trail to the same pastel palette
   お兄ちゃんはおしまい！
   ===================================================== */
(function () {
  'use strict';

  /* ─────────────────────────────────────────────────
     Round-7: background video / gif
  ───────────────────────────────────────────────── */
  function mountMedia() {
    if (document.querySelector('.bgvid')) return;
    var assets = window.CATSIUM_ASSETS || {};
    var hasMp4 = !!assets.background_mp4;
    var hasGif = !!assets.background_gif;
    if (!hasMp4 && !hasGif) return;

    function addGif() {
      if (!hasGif) return;
      if (document.querySelector('img.bgvid')) return;
      var img = document.createElement('img');
      img.className = 'bgvid';
      img.alt = '';
      img.setAttribute('aria-hidden', 'true');
      img.onerror = function () { img.remove(); };
      img.src = '/static/background.gif';
      document.body.appendChild(img);
    }

    if (!hasMp4) {
      addGif();
      return;
    }
    var v = document.createElement('video');
    v.className  = 'bgvid';
    v.autoplay   = true;
    v.loop       = true;
    v.muted      = true;
    v.playsInline = true;
    v.setAttribute('playsinline', '');
    v.setAttribute('aria-hidden', 'true');
    v.onerror = function () { v.remove(); addGif(); };
    v.src = '/static/background.mp4';
    document.body.appendChild(v);
    var p = v.play && v.play();
    if (p && p.catch) p.catch(function () {});
  }

  /* ─────────────────────────────────────────────────
     Round-9: Onimai decoration layer
     Guard: skip everything if user prefers reduced motion
  ───────────────────────────────────────────────── */
  var prefersReduced =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── 1. Sakura petal drift + quiet star twinkle ──
     One canvas, two very small particle systems:
       - Petal: drifts down, gentle sway, slow rotation. Pastel
         pink/white only, low opacity, no glow — meant to be
         glanced past, not stared at.
       - Star: near-fixed point that fades in/out slowly. Sparse
         (single digits) so it never reads as "snow".
  ── */
  function initSakuraCanvas() {
    var canvas = document.createElement('canvas');
    canvas.id = 'onimai-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    document.body.appendChild(canvas);

    var ctx = canvas.getContext('2d');
    var W = 0, H = 0;

    function resize() {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize, { passive: true });

    // Pastel-only palette — no rainbow, just sakura tones.
    var PETAL_COLORS = ['#ffd6ea', '#ffb8d9', '#ffffff', '#f3d9ff'];
    var STAR_COLOR    = '#fff7e6';

    /* ── Petal ── */
    function Petal(init) { this.reset(!!init); }

    Petal.prototype.reset = function (init) {
      this.x        = Math.random() * W;
      this.y        = init ? Math.random() * H : -16;
      this.r        = 2.5 + Math.random() * 3.5;      // smaller than the old snow
      this.speedY   = 0.16 + Math.random() * 0.34;     // slower, more graceful fall
      this.driftX   = (Math.random() - 0.5) * 0.25;
      this.wave     = Math.random() * Math.PI * 2;
      this.waveAmp  = 0.3 + Math.random() * 0.55;
      this.waveFreq = 0.005 + Math.random() * 0.008;
      this.opacity  = 0.12 + Math.random() * 0.22;      // softer than before
      this.color    = PETAL_COLORS[Math.floor(Math.random() * PETAL_COLORS.length)];
      this.rot      = Math.random() * Math.PI * 2;
      this.rotSpd   = (Math.random() - 0.5) * 0.014;
    };

    Petal.prototype.update = function () {
      this.wave += this.waveFreq * 60;
      this.x   += Math.sin(this.wave) * this.waveAmp + this.driftX;
      this.y   += this.speedY;
      this.rot += this.rotSpd;
      if (this.y > H + 20 || this.x < -40 || this.x > W + 40) this.reset(false);
    };

    Petal.prototype.draw = function (ctx) {
      ctx.save();
      ctx.globalAlpha = this.opacity;
      ctx.fillStyle   = this.color;
      ctx.translate(this.x, this.y);
      ctx.rotate(this.rot);

      // single sakura-petal shape: a soft rounded teardrop with a
      // small notch at the tip, built from two bezier curves.
      var r = this.r;
      ctx.beginPath();
      ctx.moveTo(0, -r * 1.3);
      ctx.bezierCurveTo(r * 1.05, -r * 0.9, r * 0.85, r * 0.85, 0, r * 1.1);
      ctx.bezierCurveTo(-r * 0.85, r * 0.85, -r * 1.05, -r * 0.9, 0, -r * 1.3);
      ctx.fill();

      ctx.restore();
    };

    /* ── Star ── */
    function Star() {
      this.x       = Math.random() * W;
      this.y       = Math.random() * H * 0.7;   // keep them in the upper field
      this.r       = 0.8 + Math.random() * 1.4;
      this.phase   = Math.random() * Math.PI * 2;
      this.speed   = 0.004 + Math.random() * 0.006;
      this.driftX  = (Math.random() - 0.5) * 0.04;
      this.driftY  = (Math.random() - 0.5) * 0.04;
    }

    Star.prototype.update = function () {
      this.phase += this.speed * 60;
      this.x += this.driftX;
      this.y += this.driftY;
      if (this.x < 0) this.x = W; if (this.x > W) this.x = 0;
      if (this.y < 0) this.y = 0; if (this.y > H * 0.7) this.y = H * 0.7;
    };

    Star.prototype.draw = function (ctx) {
      var twinkle = (Math.sin(this.phase) + 1) / 2;     // 0..1
      var opacity = 0.08 + twinkle * 0.34;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = STAR_COLOR;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    };

    // Small pools on purpose — calm, not a blizzard.
    var PETAL_COUNT = 20;
    var STAR_COUNT  = 9;
    var petals = [];
    var stars   = [];
    for (var i = 0; i < PETAL_COUNT; i++) petals.push(new Petal(true));
    for (var j = 0; j < STAR_COUNT; j++)   stars.push(new Star());

    function animate() {
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < stars.length; i++) { stars[i].update(); stars[i].draw(ctx); }
      for (var j = 0; j < petals.length; j++) { petals[j].update(); petals[j].draw(ctx); }
      requestAnimationFrame(animate);
    }
    animate();
  }

  /* ── 2. Cursor sparkle trail (pastel palette) ── */
  function initCursorTrail() {
    var COLORS  = ['#ffb8d9', '#ffd6ea', '#f3d9ff', '#ffffff'];
    var lastT   = 0;
    var lastX   = -999;
    var lastY   = -999;

    function spawnSparkle(x, y) {
      var el   = document.createElement('div');
      el.className = 'cat-sparkle';
      var sz   = 3 + Math.random() * 5;
      var col  = COLORS[Math.floor(Math.random() * COLORS.length)];
      var ang  = Math.random() * Math.PI * 2;
      var dist = 14 + Math.random() * 26;
      var tx   = (Math.cos(ang) * dist).toFixed(1) + 'px';
      var ty   = (Math.sin(ang) * dist).toFixed(1) + 'px';

      el.style.cssText = [
        'left:'        + x + 'px',
        'top:'         + y + 'px',
        'width:'       + sz + 'px',
        'height:'      + sz + 'px',
        'margin-left:' + (-sz / 2).toFixed(1) + 'px',
        'margin-top:'  + (-sz / 2).toFixed(1) + 'px',
        'background:'  + col,
        'box-shadow:0 0 ' + (sz * 1.3).toFixed(1) + 'px ' + col,
        '--tx:'        + tx,
        '--ty:'        + ty
      ].join(';');

      document.body.appendChild(el);
      setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
      }, 700);
    }

    document.addEventListener('mousemove', function (e) {
      var now = Date.now();
      if (now - lastT < 45) return;           // ~22 fps max
      var dx = e.clientX - lastX;
      var dy = e.clientY - lastY;
      if (dx * dx + dy * dy < 64) return;     // need ≥8px movement
      lastT = now;
      lastX = e.clientX;
      lastY = e.clientY;
      spawnSparkle(e.clientX, e.clientY);
      if (Math.random() < 0.3) spawnSparkle(e.clientX, e.clientY);
    }, { passive: true });
  }

  /* ── Master Onimai init ──
     Note: the round-8 floating-emoji layer (initFloatingDecos)
     has been removed per feedback that it was distracting. ── */
  function initOnimai() {
    if (prefersReduced) return;
    initSakuraCanvas();
    initCursorTrail();
  }

  /* ─────────────────────────────────────────────────
     Public API + auto-mount
  ───────────────────────────────────────────────── */
  window.CatsiumShell = {
    mount: function () { mountMedia(); }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      window.CatsiumShell.mount();
      initOnimai();
    });
  } else {
    window.CatsiumShell.mount();
    initOnimai();
  }
})();
