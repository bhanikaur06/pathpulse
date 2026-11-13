/* Path Pulse — Appearance card on the settings page (light / dark / match device) */
(function () {
  "use strict";
  var KEY = "pp_theme";
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function getChoice() {
    try { return localStorage.getItem(KEY) || "light"; } catch (e) { return "light"; }
  }
  function resolve(choice) {
    if (choice === "system") return media && media.matches ? "dark" : "light";
    return choice === "dark" ? "dark" : "light";
  }
  function apply(choice) {
    try { localStorage.setItem(KEY, choice); } catch (e) {}
    document.documentElement.setAttribute("data-theme", resolve(choice));
  }

  if (media && media.addEventListener) {
    media.addEventListener("change", function () { if (getChoice() === "system") apply("system"); });
  }

  function buildCard() {
    if (location.pathname.replace(/\/+$/, "") !== "/settings") return;
    var main = document.querySelector(".main-area");
    if (!main || document.getElementById("themeCard")) return;

    var card = document.createElement("div");
    card.className = "card";
    card.id = "themeCard";
    card.innerHTML =
      '<div class="card-title">🎨 Appearance</div>' +
      '<p style="font-size:0.88rem; margin:0 0 14px;">Choose how Path Pulse looks on this device.</p>' +
      '<div class="theme-picker" role="radiogroup" aria-label="Theme"></div>';
    var picker = card.querySelector(".theme-picker");

    function refresh() {
      var c = getChoice();
      Array.prototype.forEach.call(picker.children, function (b) {
        b.setAttribute("aria-checked", b.dataset.choice === c ? "true" : "false");
      });
    }

    [["light", "Light"], ["dark", "Dark"], ["system", "Match device"]].forEach(function (o) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "theme-option";
      b.setAttribute("role", "radio");
      b.dataset.choice = o[0];
      b.innerHTML = '<span class="theme-swatch swatch-' + o[0] + '"></span><span>' + o[1] + "</span>";
      b.addEventListener("click", function () { apply(o[0]); refresh(); });
      picker.appendChild(b);
    });
    refresh();

    var grids = main.querySelectorAll(".grid");
    (grids.length ? grids[grids.length - 1] : main).appendChild(card);
  }

  apply(getChoice());
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", buildCard);
  else buildCard();
})();
