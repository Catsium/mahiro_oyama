/* CATSIUM 98 desktop shell — window manager, taskbar, boot, dialogs, sounds.
   Used only by base98.html (dashboard + bot pages). Boot/clock/start-menu
   patterns harvested from the retired crt-shell.js. */
(function () {
  'use strict';

  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var WM = window.matchMedia('(min-width: 768px)').matches; /* desktop window-manager mode */
  var LS_LAYOUT = 'win98:' + location.pathname;
  var LS_CRT = 'win98:crt';
  var LS_SND = 'win98:snd';

  var desktop = document.getElementById('desktop');
  var wins = Array.prototype.slice.call(document.querySelectorAll('.win98-window[data-win]'));
  var zTop = 10;

  /* ---------------- persisted layout state ---------------- */
  var state = {};
  try { state = JSON.parse(localStorage.getItem(LS_LAYOUT) || '{}') || {}; } catch (e) { state = {}; }
  function saveState() {
    try { localStorage.setItem(LS_LAYOUT, JSON.stringify(state)); } catch (e) {}
  }
  function st(slug) {
    if (!state[slug]) state[slug] = {};
    return state[slug];
  }

  /* ---------------- 8-bit sounds (synth, off by default) ---------------- */
  var sndOn = localStorage.getItem(LS_SND) === '1';
  var actx = null;
  function beep(freq, dur, type, vol, when) {
    if (!sndOn) return;
    try {
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      var t = actx.currentTime + (when || 0);
      var o = actx.createOscillator();
      var g = actx.createGain();
      o.type = type || 'square';
      o.frequency.value = freq;
      g.gain.setValueAtTime(vol || 0.04, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g); g.connect(actx.destination);
      o.start(t); o.stop(t + dur + 0.02);
    } catch (e) {}
  }
  var snd = {
    tick:  function () { beep(880, 0.04); },
    open:  function () { beep(523, 0.07); beep(784, 0.09, 'square', 0.04, 0.06); },
    close: function () { beep(784, 0.07); beep(392, 0.09, 'square', 0.04, 0.06); },
    ding:  function () { beep(1046, 0.18, 'triangle', 0.05); },
    chime: function () { beep(523, 0.12, 'triangle', 0.05); beep(659, 0.12, 'triangle', 0.05, 0.1); beep(784, 0.2, 'triangle', 0.05, 0.2); }
  };
  function syncSndUi() {
    var b = document.getElementById('sndToggle');
    if (b) b.textContent = sndOn ? '🔊' : '🔇';
    var c = document.getElementById('sndChk');
    if (c) c.checked = sndOn;
  }
  function setSnd(on) {
    sndOn = on;
    try { localStorage.setItem(LS_SND, on ? '1' : '0'); } catch (e) {}
    syncSndUi();
    if (on) snd.ding();
  }

  /* ---------------- CRT overlay ---------------- */
  function setCrt(on) {
    document.body.classList.toggle('crt-on', on);
    try { localStorage.setItem(LS_CRT, on ? '1' : '0'); } catch (e) {}
    var c = document.getElementById('crtChk');
    if (c) c.checked = on;
  }

  /* ---------------- window manager ---------------- */
  function raise(el) {
    zTop += 1;
    el.style.zIndex = zTop;
    wins.forEach(function (w) { w.classList.toggle('focused', w === el); });
  }

  function kick() {
    /* hidden canvases / range slider need a resize nudge after reopen */
    requestAnimationFrame(function () { window.dispatchEvent(new Event('resize')); });
  }

  function openWin(el) {
    el.classList.remove('hidden-win');
    st(el.dataset.win).open = true;
    st(el.dataset.win).min = false;
    saveState(); raise(el); syncTaskbar(); kick(); snd.open();
  }
  function closeWin(el) {
    if (WM) {
      el.classList.add('hidden-win');
      st(el.dataset.win).open = false;
      saveState(); syncTaskbar(); snd.close();
    } else {
      el.classList.toggle('collapsed'); /* mobile: X collapses like minimize */
    }
  }
  function minWin(el) {
    if (WM) {
      el.classList.add('hidden-win');
      st(el.dataset.win).open = true;
      st(el.dataset.win).min = true;
      saveState(); syncTaskbar(); snd.close();
    } else {
      el.classList.toggle('collapsed');
    }
  }
  function restoreWin(el) {
    el.classList.remove('hidden-win');
    st(el.dataset.win).min = false;
    saveState(); raise(el); syncTaskbar(); kick(); snd.open();
  }
  function maxWin(el) {
    el.classList.toggle('maxed');
    st(el.dataset.win).max = el.classList.contains('maxed');
    saveState(); raise(el); kick(); snd.tick();
  }

  function applyGeometry(el) {
    var s = st(el.dataset.win);
    var w = parseInt(el.dataset.w, 10) || 480;
    var x = (s.x != null) ? s.x : parseInt(el.dataset.x, 10) || 16;
    var y = (s.y != null) ? s.y : parseInt(el.dataset.y, 10) || 16;
    var h = parseInt(el.dataset.h, 10) || 0;
    var dw = desktop.clientWidth, dh = desktop.clientHeight;
    x = Math.max(-w + 120, Math.min(x, dw - 60));
    y = Math.max(0, Math.min(y, dh - 40));
    if (w > dw - 16) w = dw - 16;
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    el.style.width = w + 'px';
    if (h) { el.style.height = Math.min(h, dh - 16) + 'px'; }
    else { el.style.maxHeight = Math.max(160, dh - y - 8) + 'px'; }
    if (s.max) el.classList.add('maxed');
    if (s.open === false || (s.open == null && el.dataset.closed)) el.classList.add('hidden-win');
    else if (s.open) el.classList.remove('hidden-win');
    if (s.min) el.classList.add('hidden-win');
  }

  function drag(el) {
    var bar = el.querySelector('.titlebar');
    if (!bar) return;
    var sx, sy, ox, oy, moving = false;
    bar.addEventListener('pointerdown', function (e) {
      if (e.target.closest('.t-controls')) return;
      if (el.classList.contains('maxed')) return;
      moving = true;
      sx = e.clientX; sy = e.clientY;
      ox = el.offsetLeft; oy = el.offsetTop;
      el.classList.add('dragging');
      bar.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    bar.addEventListener('pointermove', function (e) {
      if (!moving) return;
      var w = el.offsetWidth;
      var x = ox + (e.clientX - sx);
      var y = oy + (e.clientY - sy);
      x = Math.max(-w + 120, Math.min(x, desktop.clientWidth - 60));
      y = Math.max(0, Math.min(y, desktop.clientHeight - 30));
      el.style.left = x + 'px';
      el.style.top = y + 'px';
    });
    bar.addEventListener('pointerup', function () {
      if (!moving) return;
      moving = false;
      el.classList.remove('dragging');
      var s = st(el.dataset.win);
      s.x = el.offsetLeft; s.y = el.offsetTop;
      saveState();
    });
  }

  function windowControls(el) {
    el.querySelectorAll('.t-controls button').forEach(function (b) {
      b.addEventListener('click', function (e) {
        e.preventDefault(); e.stopPropagation();
        if (b.dataset.act === 'min') minWin(el);
        else if (b.dataset.act === 'max') maxWin(el);
        else closeWin(el);
      });
    });
    el.addEventListener('pointerdown', function () { if (WM) raise(el); });
  }

  /* ---------------- taskbar + desktop icons ---------------- */
  function iconSvg(el) {
    var ic = el.querySelector('.titlebar .t-icon');
    return ic ? ic.innerHTML : '';
  }

  function syncTaskbar() {
    var bar = document.getElementById('tbWindows');
    if (!bar) return;
    bar.innerHTML = '';
    if (!WM) return;
    wins.forEach(function (el) {
      var s = st(el.dataset.win);
      var open = !(s.open === false || (s.open == null && el.dataset.closed));
      if (!open) return;
      var minimized = !!s.min;
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'tb-win' + (!minimized && el.classList.contains('focused') ? ' active' : '');
      b.innerHTML = '<span class="t-icon">' + iconSvg(el) + '</span><span class="lbl"></span>';
      b.querySelector('.lbl').textContent = el.dataset.title;
      b.addEventListener('click', function () {
        if (el.classList.contains('hidden-win')) restoreWin(el);
        else if (el.classList.contains('focused')) minWin(el);
        else raise(el);
        syncTaskbar();
      });
      bar.appendChild(b);
    });
  }

  function deskIcons() {
    var box = document.getElementById('deskIcons');
    if (!box || !WM) return;
    wins.forEach(function (el) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'desk-icon';
      b.innerHTML = '<span class="di-img">' + iconSvg(el) + '</span><span class="di-lbl"></span>';
      b.querySelector('.di-lbl').textContent = el.dataset.title;
      b.addEventListener('dblclick', function () { openWin(el); });
      b.addEventListener('click', function () {
        if (!el.classList.contains('hidden-win')) raise(el);
        syncTaskbar();
      });
      box.appendChild(b);
    });
  }

  /* ---------------- boot sequence (once per tab session) ---------------- */
  function boot() {
    if (REDUCED || sessionStorage.getItem('catsium_booted')) return;
    sessionStorage.setItem('catsium_booted', '1');
    var el = document.createElement('div');
    el.id = 'boot';
    el.setAttribute('aria-hidden', 'true');
    el.innerHTML =
      '<div class="boot-logo">CATSIUM <span class="accent">98</span>' +
      '<small>MAHIRO OYAMA TRADING SYSTEM v1.998</small></div>' +
      '<div class="boot-bar"><div class="boot-fill"></div></div>' +
      '<div class="boot-log">' +
      '<div>&gt; BIOS check: 640K RAM ........ <span class="ok">OK</span></div>' +
      '<div>&gt; loading market feed ......... <span class="ok">OK</span></div>' +
      '<div>&gt; waking trading bot .......... <span class="ok">OK</span></div>' +
      '<div>&gt; petting cat ................. <span class="ok">OK</span></div>' +
      '</div>' +
      '<div class="boot-skip">CLICK TO SKIP</div>';
    document.body.appendChild(el);
    var killed = false;
    function finish() {
      if (killed) return;
      killed = true;
      el.classList.add('done');
      setTimeout(function () { el.remove(); }, 320);
      snd.chime();
    }
    el.addEventListener('click', finish);
    setTimeout(finish, 2100);
  }

  /* ---------------- taskbar clock ---------------- */
  function clock() {
    var c = document.getElementById('tbClock');
    if (!c) return;
    var tick = function () {
      c.textContent = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    };
    tick();
    setInterval(tick, 30 * 1000);
  }

  /* ---------------- start menu ---------------- */
  function startMenu() {
    var btn = document.getElementById('startBtn');
    var menu = document.getElementById('startMenu');
    if (!btn || !menu) return;
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = menu.classList.toggle('open');
      btn.classList.toggle('open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      snd.tick();
    });
    document.addEventListener('click', function () {
      menu.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
    });
    menu.addEventListener('click', function (e) { e.stopPropagation(); });
    menu.querySelectorAll('[data-menu]').forEach(function (item) {
      item.addEventListener('click', function () {
        var act = item.dataset.menu;
        menu.classList.remove('open');
        btn.classList.remove('open');
        if (act === 'display') {
          var dp = document.querySelector('[data-win="display"]');
          if (dp) openWin(dp);
        } else if (act === 'reset-layout') {
          state = {};
          try { localStorage.removeItem(LS_LAYOUT); } catch (e) {}
          wins.forEach(function (el) {
            el.classList.remove('maxed', 'hidden-win');
            if (el.dataset.closed) el.classList.add('hidden-win');
            if (WM) applyGeometry(el);
          });
          syncTaskbar(); kick(); snd.open();
        }
      });
    });
  }

  /* ---------------- dialogs: flash close + data-confirm ---------------- */
  function dlgLayer() { return document.getElementById('dlgLayer'); }
  function dlgSync() {
    var layer = dlgLayer();
    if (layer) layer.classList.toggle('empty', !layer.querySelector('.w98-dialog'));
  }
  function dialogs() {
    var layer = dlgLayer();
    if (!layer) return;
    layer.addEventListener('click', function (e) {
      if (e.target.closest('[data-dlg-close]')) {
        var d = e.target.closest('.w98-dialog');
        if (d) d.remove();
        dlgSync(); snd.tick();
      }
    });
    if (layer.querySelector('.w98-dialog')) snd.ding();
  }

  function confirmModal() {
    document.addEventListener('submit', function (e) {
      var form = e.target;
      if (!form.matches || !form.matches('form[data-confirm]') || form.dataset.confirmed) return;
      e.preventDefault();
      snd.ding();
      var layer = dlgLayer();
      if (!layer) { form.submit(); return; }
      var d = document.createElement('div');
      d.className = 'w98-dialog';
      d.setAttribute('role', 'alertdialog');
      d.innerHTML =
        '<div class="titlebar"><span class="t-text">CONFIRM</span>' +
        '<span class="t-controls"><button type="button" data-cancel aria-label="cancel">×</button></span></div>' +
        '<div class="dlg-body"><span class="dlg-ico">' +
        '<svg viewBox="0 0 16 16" shape-rendering="crispEdges"><rect x="2" y="2" width="12" height="12" fill="#28509e" stroke="#211d17"/><rect x="5" y="4" width="6" height="2" fill="#fffdf6"/><rect x="9" y="6" width="2" height="2" fill="#fffdf6"/><rect x="7" y="8" width="2" height="2" fill="#fffdf6"/><rect x="7" y="11" width="2" height="1" fill="#fffdf6"/></svg>' +
        '</span><span class="dlg-msg"></span></div>' +
        '<div class="dlg-btns"><button type="button" class="btn" data-ok>OK</button>' +
        '<button type="button" class="btn" data-cancel>Cancel</button></div>';
      d.querySelector('.dlg-msg').textContent = form.dataset.confirm;
      layer.appendChild(d);
      layer.classList.remove('empty');
      d.querySelector('[data-ok]').addEventListener('click', function () {
        d.remove(); dlgSync(); snd.tick();
        form.dataset.confirmed = '1';
        if (form.requestSubmit) form.requestSubmit(); else form.submit();
      });
      d.querySelectorAll('[data-cancel]').forEach(function (b) {
        b.addEventListener('click', function () { d.remove(); dlgSync(); snd.tick(); });
      });
      d.querySelector('[data-ok]').focus();
    });
  }

  /* ---------------- display properties bindings ---------------- */
  function displayProps() {
    var crtChk = document.getElementById('crtChk');
    if (crtChk) crtChk.addEventListener('change', function () { setCrt(crtChk.checked); snd.tick(); });
    var sndChk = document.getElementById('sndChk');
    if (sndChk) sndChk.addEventListener('change', function () { setSnd(sndChk.checked); });
    var sndBtn = document.getElementById('sndToggle');
    if (sndBtn) sndBtn.addEventListener('click', function () { setSnd(!sndOn); });
  }

  /* ---------------- generic click blips on beveled controls ---------------- */
  function clickSounds() {
    document.addEventListener('click', function (e) {
      if (e.target.closest('.btn,.ghost-btn,.add-btn,.range-btn,.tb-win,.desk-icon,.pc-day')) snd.tick();
    }, true);
  }

  /* ---------------- mount ---------------- */
  setCrt(localStorage.getItem(LS_CRT) === '1');
  syncSndUi();
  boot();
  clock();
  startMenu();
  dialogs();
  confirmModal();
  displayProps();
  clickSounds();

  wins.forEach(windowControls);
  if (WM) {
    document.body.classList.add('wm');
    wins.forEach(function (el, i) {
      applyGeometry(el);
      drag(el);
      /* initial stacking: DOM order, data-behind sinks to the bottom */
      el.style.zIndex = el.dataset.behind ? 9 : (zTop = 10 + i);
    });
    deskIcons();
    syncTaskbar();
    var front = document.querySelector('.win98-window[data-front]') || wins[0];
    if (front) raise(front);
    window.addEventListener('resize', function () {
      /* keep windows reachable after viewport shrink */
      wins.forEach(function (el) {
        if (el.classList.contains('hidden-win') || el.classList.contains('maxed')) return;
        var x = Math.max(-el.offsetWidth + 120, Math.min(el.offsetLeft, desktop.clientWidth - 60));
        var y = Math.max(0, Math.min(el.offsetTop, desktop.clientHeight - 30));
        el.style.left = x + 'px'; el.style.top = y + 'px';
      });
    });
  }
})();
