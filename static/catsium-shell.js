/* =====================================================
   CATSIUM shared shell
   - Full-screen media background fallback.
   - Onimai-inspired pastel sakura motion layer.
   - No ambient emoji.
   ===================================================== */
(function () {
  'use strict';

  var motionStarted = false;

  function mountMedia() {
    if (document.querySelector('.bgvid')) return;

    function addGif() {
      if (document.querySelector('img.bgvid')) return;
      var img = document.createElement('img');
      img.className = 'bgvid';
      img.alt = '';
      img.setAttribute('aria-hidden', 'true');
      img.onerror = function () { img.remove(); };
      img.src = '/static/background.gif';
      document.body.appendChild(img);
    }

    var video = document.createElement('video');
    video.className = 'bgvid';
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('aria-hidden', 'true');
    video.onerror = function () { video.remove(); addGif(); };
    video.src = '/static/background.mp4';
    document.body.appendChild(video);

    var playPromise = video.play && video.play();
    if (playPromise && playPromise.catch) playPromise.catch(function () {});
  }

  function initSakuraCanvas() {
    if (document.getElementById('onimai-canvas')) return;

    var canvas = document.createElement('canvas');
    canvas.id = 'onimai-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    document.body.appendChild(canvas);

    var ctx = canvas.getContext('2d', { alpha: true });
    var width = 0;
    var height = 0;
    var dpr = 1;
    var petals = [];
    var running = true;
    var raf = 0;
    var lastFrame = 0;
    var colors = ['#ffd4e4', '#ffc1da', '#f6a8c8', '#f9ddea', '#e8dcff', '#fff7fb'];

    function targetCount() {
      return Math.max(16, Math.min(30, Math.round((width * height) / 56000)));
    }

    function syncPool() {
      var wanted = targetCount();
      while (petals.length < wanted) petals.push(new Petal(true));
      while (petals.length > wanted) petals.pop();
    }

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth || document.documentElement.clientWidth || 1;
      height = window.innerHeight || document.documentElement.clientHeight || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      syncPool();
    }

    function Petal(initial) {
      this.reset(initial);
    }

    Petal.prototype.reset = function (initial) {
      this.x = Math.random() * width;
      this.y = initial ? Math.random() * height : -36 - Math.random() * 90;
      this.size = 5 + Math.random() * 7;
      this.speedY = 0.16 + Math.random() * 0.34;
      this.driftX = (0.06 + Math.random() * 0.18) * (Math.random() < 0.5 ? -1 : 1);
      this.sway = Math.random() * Math.PI * 2;
      this.swaySpeed = 0.012 + Math.random() * 0.015;
      this.swayAmp = 0.28 + Math.random() * 0.85;
      this.opacity = 0.08 + Math.random() * 0.16;
      this.color = colors[Math.floor(Math.random() * colors.length)];
      this.rotation = Math.random() * Math.PI * 2;
      this.rotationSpeed = (Math.random() - 0.5) * 0.032;
      this.depth = 0.72 + Math.random() * 0.42;
    };

    Petal.prototype.update = function (dt) {
      var step = Math.min(dt, 34) / 16.67;
      this.sway += this.swaySpeed * step;
      this.x += (this.driftX + Math.sin(this.sway) * this.swayAmp) * step;
      this.y += this.speedY * this.depth * step;
      this.rotation += this.rotationSpeed * step;

      if (this.y > height + 40 || this.x < -70 || this.x > width + 70) {
        this.reset(false);
      }
    };

    Petal.prototype.draw = function () {
      var s = this.size;
      ctx.save();
      ctx.globalAlpha = this.opacity;
      ctx.translate(this.x, this.y);
      ctx.rotate(this.rotation);
      ctx.fillStyle = this.color;
      ctx.shadowColor = this.color;
      ctx.shadowBlur = 3;

      ctx.beginPath();
      ctx.moveTo(0, -s);
      ctx.bezierCurveTo(s * 0.62, -s * 0.55, s * 0.72, s * 0.25, 0, s);
      ctx.bezierCurveTo(-s * 0.5, s * 0.18, -s * 0.6, -s * 0.55, 0, -s);
      ctx.fill();

      ctx.globalAlpha = this.opacity * 0.55;
      ctx.strokeStyle = '#fff7fb';
      ctx.lineWidth = 0.55;
      ctx.beginPath();
      ctx.moveTo(0, -s * 0.52);
      ctx.quadraticCurveTo(s * 0.14, 0, 0, s * 0.56);
      ctx.stroke();
      ctx.restore();
    };

    function frame(now) {
      if (!running) {
        raf = 0;
        return;
      }

      var dt = lastFrame ? now - lastFrame : 16.67;
      lastFrame = now;
      ctx.clearRect(0, 0, width, height);
      for (var i = 0; i < petals.length; i++) {
        petals[i].update(dt);
        petals[i].draw();
      }
      raf = window.requestAnimationFrame(frame);
    }

    function start() {
      if (raf) return;
      running = true;
      lastFrame = 0;
      raf = window.requestAnimationFrame(frame);
    }

    function stop() {
      running = false;
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
    }

    resize();
    window.addEventListener('resize', resize, { passive: true });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stop();
      else start();
    });
    start();
  }

  function spawnCursorPetal(x, y, burst) {
    var el = document.createElement('div');
    var colors = ['#ffd4e4', '#ffc1da', '#f6a8c8', '#e8dcff', '#fff7fb'];
    var size = (burst ? 4 : 3) + Math.random() * (burst ? 6 : 4);
    var angle = Math.random() * Math.PI * 2;
    var distance = (burst ? 20 : 12) + Math.random() * (burst ? 34 : 22);
    var color = colors[Math.floor(Math.random() * colors.length)];

    el.className = 'cat-sparkle';
    el.style.cssText = [
      'left:' + x + 'px',
      'top:' + y + 'px',
      'width:' + size.toFixed(1) + 'px',
      'height:' + (size * 1.45).toFixed(1) + 'px',
      'margin-left:' + (-size / 2).toFixed(1) + 'px',
      'margin-top:' + (-size / 2).toFixed(1) + 'px',
      'background:' + color,
      'box-shadow:0 0 ' + (size * 1.5).toFixed(1) + 'px ' + color,
      '--tx:' + (Math.cos(angle) * distance).toFixed(1) + 'px',
      '--ty:' + (Math.sin(angle) * distance).toFixed(1) + 'px',
      '--rot:' + Math.round(Math.random() * 180) + 'deg'
    ].join(';');

    document.body.appendChild(el);
    window.setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 720);
  }

  function initCursorTrail() {
    if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;

    var lastTime = 0;
    var lastX = -999;
    var lastY = -999;

    document.addEventListener('mousemove', function (event) {
      var now = Date.now();
      var dx = event.clientX - lastX;
      var dy = event.clientY - lastY;
      if (now - lastTime < 70 || dx * dx + dy * dy < 144) return;

      lastTime = now;
      lastX = event.clientX;
      lastY = event.clientY;
      spawnCursorPetal(event.clientX, event.clientY, false);
    }, { passive: true });
  }

  function initPetalBursts() {
    var selector = '.btn,.ghost-btn,.add-btn,.range-btn,.stock-card';
    document.addEventListener('click', function (event) {
      if (!event.target.closest || !event.target.closest(selector)) return;
      for (var i = 0; i < 5; i++) {
        window.setTimeout(function () {
          spawnCursorPetal(event.clientX, event.clientY, true);
        }, i * 18);
      }
    }, true);
  }

  function initOnimaiMotion() {
    if (motionStarted) return;
    motionStarted = true;
    document.documentElement.classList.add('motion-ready');
    initSakuraCanvas();
    initCursorTrail();
    initPetalBursts();
  }

  window.CatsiumShell = {
    mount: function () {
      mountMedia();
      initOnimaiMotion();
    },
    refreshMotion: initOnimaiMotion
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', window.CatsiumShell.mount);
  } else {
    window.CatsiumShell.mount();
  }
})();
