/* Path Pulse UI enhancements: hold-to-send SOS, mobile SOS button,
   status hero, setup checklist, readable place names. */
(function () {
  "use strict";
  var HOLD_MS = 2000;
  var HINT = "Hold the SOS button for 2 seconds to send an alert.";

  function say(msg) { if (typeof window.toast === "function") window.toast(msg); }
  function safe(fn) { try { fn(); } catch (e) { console.warn("[ui]", e); } }

  /* ---------- 1 + 5: hold-to-send SOS (desktop button + mobile floating button) ---------- */

  var allowSosClick = false;
  var lastFired = 0;

  function attachHold(el, onDone) {
    var start = 0, raf = 0, active = false;
    var original = el.innerHTML;

    function reset() {
      active = false;
      cancelAnimationFrame(raf);
      el.classList.remove("is-holding");
      el.style.setProperty("--p", "0");
      el.innerHTML = original;
    }
    function frame(now) {
      if (!active) return;
      var p = Math.min((now - start) / HOLD_MS, 1);
      el.style.setProperty("--p", p.toFixed(3));
      if (p >= 1) {
        reset();
        if (navigator.vibrate) navigator.vibrate(200);
        onDone();
        return;
      }
      raf = requestAnimationFrame(frame);
    }
    function begin(e) {
      if (active) return;
      if (e && e.type === "pointerdown" && e.button !== 0) return;
      original = el.innerHTML;
      active = true;
      start = performance.now();
      el.classList.add("is-holding");
      if (el.dataset.holdText) el.textContent = el.dataset.holdText;
      raf = requestAnimationFrame(frame);
    }
    function cancel() { if (active) reset(); }

    el.addEventListener("pointerdown", begin);
    ["pointerup", "pointerleave", "pointercancel"].forEach(function (t) {
      el.addEventListener(t, cancel);
    });
    el.addEventListener("keydown", function (e) {
      if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); begin(); }
    });
    el.addEventListener("keyup", function (e) {
      if (e.key === " " || e.key === "Enter") cancel();
    });
    el.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  }

  function setupSosHold() {
    var btn = document.getElementById("manualSosBtn");
    if (!btn) return;

    function fire() {
      lastFired = Date.now();
      allowSosClick = true;
      try { btn.click(); } finally { allowSosClick = false; }   // runs your existing SOS + photo code
    }

    // Block ordinary single clicks on the SOS button (capture phase runs before the old handler)
    document.addEventListener("click", function (e) {
      if (allowSosClick) return;
      var hit = e.target.closest && e.target.closest("#manualSosBtn, .sos-fab");
      if (!hit) return;
      e.stopImmediatePropagation();
      e.preventDefault();
      if (Date.now() - lastFired > 800) say(HINT);
    }, true);

    btn.classList.add("hold-btn");
    btn.dataset.holdText = "Keep holding…";
    btn.title = "Press and hold for 2 seconds";
    attachHold(btn, fire);

    var fab = document.createElement("button");
    fab.type = "button";
    fab.className = "sos-fab hold-btn";
    fab.textContent = "SOS";
    fab.setAttribute("aria-label", "Hold for 2 seconds to send SOS");
    document.body.appendChild(fab);
    attachHold(fab, fire);
  }

  /* ---------- 3: big central status pulse ---------- */

  function setupStatusHero() {
    var pulse = document.getElementById("pulseWrap");
    var main = document.querySelector(".main-area");
    if (!pulse || !main) return;
    var card = pulse.closest(".card");
    if (!card || card.classList.contains("status-hero")) return;

    var info = document.createElement("div");
    info.className = "hero-info";
    var label = document.createElement("div");
    label.className = "hero-label";
    label.textContent = "Current status";
    info.appendChild(label);

    Array.prototype.slice.call(card.children).forEach(function (child) {
      if (child === pulse) return;
      if (child.classList.contains("card-title")) { child.remove(); return; }
      info.appendChild(child);
    });
    card.appendChild(info);
    card.classList.add("status-hero");

    var topline = main.querySelector(".topline");
    main.insertBefore(card, topline ? topline.nextSibling : main.firstChild);

    setTimeout(function () {
      try { if (typeof map !== "undefined" && map && map.invalidateSize) map.invalidateSize(); } catch (e) {}
    }, 400);
  }

  /* ---------- 4: setup checklist for new users ---------- */

  function setupChecklist() {
    var main = document.querySelector(".main-area");
    if (!main || !document.getElementById("pulseWrap")) return;
    try { if (sessionStorage.getItem("pp_setup_hidden") === "1") return; } catch (e) {}

    var locCheck = (navigator.permissions && navigator.permissions.query)
      ? navigator.permissions.query({ name: "geolocation" })
          .then(function (r) { return r.state === "granted"; })
          .catch(function () { return null; })
      : Promise.resolve(null);

    var status = fetch("/setup_status")
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });

    Promise.all([status, locCheck]).then(function (res) {
      var s = res[0], loc = res[1];
      if (!s) return;
      var items = [
        { done: s.contacts > 0, text: "Add an emergency contact", href: "/contacts" },
        { done: !!s.code_word, text: "Set your secret code word", href: "/settings" },
        { done: loc !== false, text: "Allow location access", href: null },
      ];
      var doneCount = items.filter(function (i) { return i.done; }).length;
      if (doneCount === items.length) return;

      var card = document.createElement("div");
      card.className = "card setup-card";
      card.innerHTML =
        '<div class="setup-head"><div class="card-title" style="margin-bottom:0;">Get set up</div>' +
        '<span class="setup-count">' + doneCount + " of " + items.length + " done</span>" +
        '<button type="button" class="btn btn-ghost btn-sm setup-hide">Hide</button></div>' +
        '<div class="setup-list"></div>';
      var list = card.querySelector(".setup-list");

      items.forEach(function (it) {
        var el = document.createElement(it.href && !it.done ? "a" : "div");
        el.className = "setup-item" + (it.done ? " done" : "");
        if (it.href && !it.done) el.href = it.href;
        if (!it.href && !it.done) el.title = "Click the icon left of the address bar and allow Location";
        el.innerHTML = '<span class="tick">' + (it.done ? "✓" : "") + '</span><span class="label">' + it.text + "</span>";
        list.appendChild(el);
      });

      card.querySelector(".setup-hide").addEventListener("click", function () {
        card.remove();
        try { sessionStorage.setItem("pp_setup_hidden", "1"); } catch (e) {}
      });

      var topline = main.querySelector(".topline");
      main.insertBefore(card, topline ? topline.nextSibling : main.firstChild);
    });
  }

  /* ---------- 2: place names instead of coordinates (OpenStreetMap Nominatim) ---------- */

  var COORD_RE = /^\s*(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\s*$/;
  var MAX_LOOKUPS = 30;

  function readCache(key) { try { return localStorage.getItem("pp_place:" + key); } catch (e) { return null; } }
  function writeCache(key, v) { try { localStorage.setItem("pp_place:" + key, v); } catch (e) {} }

  function shortName(d) {
    var a = d.address || {};
    var local = a.neighbourhood || a.suburb || a.quarter || a.road || a.village || a.hamlet;
    var city = a.city || a.town || a.city_district || a.state_district || a.county || a.state;
    var parts = [local, city].filter(function (x, i, arr) { return x && arr.indexOf(x) === i; });
    if (parts.length) return parts.join(", ");
    return d.display_name ? d.display_name.split(",").slice(0, 2).join(",").trim() : null;
  }

  function show(cells, name) {
    cells.forEach(function (td) {
      if (!td.dataset.coords) td.dataset.coords = td.textContent.trim();
      td.title = td.dataset.coords;
      td.textContent = name;
      td.classList.add("place-name");
    });
  }

  function placeNames() {
    var cells = Array.prototype.slice.call(document.querySelectorAll("td"))
      .filter(function (td) { return COORD_RE.test(td.textContent); });
    if (!cells.length) return;

    var groups = {}, order = [];
    cells.forEach(function (td) {
      var m = td.textContent.match(COORD_RE);
      var lat = parseFloat(m[1]), lon = parseFloat(m[2]);
      if (lat === 0 && lon === 0) return;
      var key = lat.toFixed(3) + "," + lon.toFixed(3);
      if (!groups[key]) { groups[key] = { lat: lat, lon: lon, cells: [] }; order.push(key); }
      groups[key].cells.push(td);
    });

    var queue = [];
    order.forEach(function (key) {
      var cached = readCache(key);
      if (cached) show(groups[key].cells, cached);
      else queue.push(key);
    });

    var i = 0;
    function next() {
      if (i >= queue.length || i >= MAX_LOOKUPS) return;
      var key = queue[i++], g = groups[key];
      fetch("https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=16&accept-language=en" +
            "&lat=" + g.lat + "&lon=" + g.lon)
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) {
          var name = d && shortName(d);
          if (name) { writeCache(key, name); show(g.cells, name); }
        })
        .catch(function () {})
        .then(function () { setTimeout(next, 1100); });   // OSM allows max 1 request/second
    }
    next();
  }

  function init() {
    safe(setupSosHold);
    safe(setupStatusHero);
    safe(setupChecklist);
    safe(placeNames);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
