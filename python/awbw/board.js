// The board as the viewer and the play page draw it: terrain, buildings and
// units on a canvas, from a game's static data (G: map, units, players,
// countries) and a frame (viewer._frame: units, owners, capture points).
// Each page inlines this file; makeBoard(G, canvas) returns the drawing API.
function makeBoard(G, cv) {
  const M = G.map, W = M.width, H = M.height, ctx = cv.getContext("2d");
  const MOVED = 1, HIDDEN = 2, CARRIED = 4, STUNNED = 8, CAPTURING = 16;
  const COUNTRY = { os: "#e8562a", bm: "#3a6fd8", ge: "#3aa155", yc: "#d9b21a", bh: "#5b4e72", rf: "#b8332e",
    gs: "#7d8a99", bd: "#9c6b3c", ab: "#f0a040", js: "#6aa86a", ci: "#3fa3c7", pc: "#e877b8", tg: "#2fa59a",
    pl: "#8a52c9", ar: "#8db33a", wn: "#b8ae94", aa: "#4f86c6", ne: "#3a3a3a", sc: "#a8a8b8", uw: "#7a5a3a" };
  const FALLBACK = ["#e8562a", "#3a6fd8"];
  const NEUTRAL = "#f1efe8";
  const TERRAIN = { plain: "#a9cf7a", mountain: "#9b7b55", wood: "#3f7f3a", river: "#63a8e0", road: "#cfc8b6",
    bridge: "#b8a27e", sea: "#2d64a8", shoal: "#e3d49a", reef: "#2d64a8", pipe: "#6e6e6e", seam: "#8e8e8e",
    rubble: "#9a8f7c", silo: "#a9cf7a", silo_empty: "#a9cf7a", teleport: "#b9b0d6" };
  const BUILDING = { city: "C", base: "B", airport: "A", port: "P", hq: "HQ", comtower: "T", lab: "L" };
  const CODES = { "Infantry": "Inf", "Mech": "Mch", "Md.Tank": "Md", "Tank": "Tnk", "Recon": "Rcn", "APC": "APC",
    "Artillery": "Art", "Rocket": "Rkt", "Anti-Air": "AA", "Missile": "Msl", "Fighter": "Ftr", "Bomber": "Bmb",
    "B-Copter": "BCp", "T-Copter": "TCp", "Battleship": "BS", "Cruiser": "Cru", "Lander": "Lnd", "Sub": "Sub",
    "Black Boat": "BB", "Carrier": "Car", "Stealth": "Stl", "Neotank": "Neo", "Piperunner": "Pip",
    "Black Bomb": "Bom", "Mega Tank": "Mg" };
  const color = (p) => COUNTRY[G.players[p].code] || FALLBACK[p];
  const light = (hex) => {  // dark text reads better on it
    const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, bl = n & 255;
    return 0.299 * r + 0.587 * g + 0.114 * bl > 165;
  };
  const textOn = (p) => (light(color(p)) ? "#2a2925" : "#fff");
  const propAt = {};
  M.props.forEach((p, i) => { propAt[p[0] + "," + p[1]] = i; });
  // exactHp: tooltips give internal HP (7.3); false gives what players see (8)
  const b = { T: 28, W, H, MOVED, HIDDEN, CARRIED, STUNNED, CAPTURING, BUILDING, color, propAt, exactHp: true };
  const hpText = (hp) => (b.exactHp ? (hp / 10).toFixed(1) : String(Math.ceil(hp / 10)));

  b.size = function (avail, max = 36) {
    const T = (b.T = Math.max(14, Math.min(max, Math.floor(avail / W))));
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * T * dpr; cv.height = H * T * dpr;
    cv.style.width = W * T + "px"; cv.style.height = H * T + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  b.kindAt = function (x, y, f) {
    const id = M.terrain[y][x];
    let k = (M.terrain_kinds[id] || ["plain"])[0];
    const p = propAt[x + "," + y];
    if (p !== undefined && k === "seam" && f.c[p] === 0) k = "rubble";
    if (p !== undefined && k === "silo" && f.c[p] === 0) k = "silo_empty";
    return k;
  };
  b.tileAt = function (e) {
    const r = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / b.T), y = Math.floor((e.clientY - r.top) / b.T);
    return x < 0 || y < 0 || x >= W || y >= H ? null : [x, y];
  };
  function drawTile(x, y, f) {
    const T = b.T, k = b.kindAt(x, y, f), px = x * T, py = y * T;
    const building = BUILDING[k];
    ctx.fillStyle = building ? TERRAIN.plain : (TERRAIN[k] || TERRAIN.plain);
    ctx.fillRect(px, py, T, T);
    if (k === "reef") {
      ctx.fillStyle = "#6f9cc0";
      for (const [a, c] of [[.25, .3], [.65, .25], [.45, .65], [.8, .7]]) ctx.fillRect(px + a * T, py + c * T, T * .12, T * .12);
    } else if (k === "mountain") {
      ctx.fillStyle = "#7d6142";
      ctx.beginPath(); ctx.moveTo(px + T * .15, py + T * .85); ctx.lineTo(px + T * .5, py + T * .2); ctx.lineTo(px + T * .85, py + T * .85); ctx.fill();
    } else if (k === "wood") {
      ctx.fillStyle = "#2f6a2c";
      ctx.beginPath(); ctx.arc(px + T / 2, py + T / 2, T * .3, 0, 7); ctx.fill();
    } else if (k === "silo" || k === "silo_empty") {
      ctx.fillStyle = k === "silo" ? "#c0392b" : "#888";
      ctx.fillRect(px + T * .35, py + T * .25, T * .3, T * .5);
    }
    if (building) {
      const p = propAt[x + "," + y];
      const owner = p === undefined ? 255 : f.o[p];
      ctx.fillStyle = owner < 2 ? color(owner) : NEUTRAL;
      ctx.strokeStyle = owner < 2 ? "rgba(0,0,0,.55)" : "rgba(0,0,0,.4)"; ctx.lineWidth = 1;
      const m = T * .1;
      if (owner >= 2) ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.roundRect(px + m, py + m, T - 2 * m, T - 2 * m, T * .12); ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = owner < 2 ? textOn(owner) : "#8b877a";
      ctx.font = `700 ${Math.round(T * (building.length > 1 ? .32 : .42))}px system-ui`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(building, px + T / 2, py + T / 2 + 1);
      if (p !== undefined && f.c[p] < 20) {  // being captured
        ctx.fillStyle = "#000"; ctx.fillRect(px + 2, py + T - 5, T - 4, 3);
        ctx.fillStyle = "#ffd23f"; ctx.fillRect(px + 2, py + T - 5, (T - 4) * f.c[p] / 20, 3);
      }
    }
    ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1;
    ctx.strokeRect(px + .5, py + .5, T - 1, T - 1);
  }
  // A unit row (viewer._frame), drawn at (x, y) if given (an animated move).
  b.drawUnit = function (u, at, dim) {
    const [, type, owner, ux, uy, hp, fuel, ammo, flags] = u;
    const T = b.T, x = at ? at[0] : ux, y = at ? at[1] : uy, px = x * T, py = y * T, m = T * .12;
    ctx.globalAlpha = (dim ?? flags & MOVED) ? .55 : 1;
    ctx.fillStyle = color(owner);
    ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.roundRect(px + m, py + m, T - 2 * m, T - 2 * m, T * .25); ctx.fill(); ctx.stroke();
    if (flags & HIDDEN) { ctx.setLineDash([2, 2]); ctx.strokeStyle = "#fff"; ctx.stroke(); ctx.setLineDash([]); }
    const name = G.units[type];
    ctx.fillStyle = textOn(owner);
    ctx.font = `700 ${Math.round(T * .3)}px system-ui`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(CODES[name] || name.slice(0, 3), px + T / 2, py + T / 2);
    ctx.globalAlpha = 1;
    const shown = Math.ceil(hp / 10);
    if (shown < 10) {
      ctx.fillStyle = "#000"; ctx.fillRect(px + T - T * .38, py + T - T * .36, T * .36, T * .34);
      ctx.fillStyle = "#fff"; ctx.font = `700 ${Math.round(T * .3)}px system-ui`;
      ctx.fillText(String(shown), px + T - T * .2, py + T - T * .18);
    }
    if (u[9] || u[10]) { ctx.fillStyle = "#ffd23f"; ctx.beginPath(); ctx.arc(px + T * .2, py + T * .2, T * .1, 0, 7); ctx.fill(); }
    if (flags & CAPTURING) { ctx.fillStyle = "#ffd23f"; ctx.fillRect(px + T * .12, py + T * .78, T * .2, T * .1); }
    const low = G.unit_stats && G.unit_stats[type];
    if (low && (fuel * 3 <= low[0] || (low[1] > 0 && ammo * 3 <= low[1]))) {  // a third or less left
      ctx.fillStyle = "#ff3b30"; ctx.font = `700 ${Math.round(T * .26)}px system-ui`;
      ctx.fillText("!", px + T * .82, py + T * .2);
    }
  };
  // Marks from the viewer: [x, y, role], role 0 the unit's start, 1 where it went, 2 what it acted on.
  b.drawMarks = function (marks) {
    const T = b.T, style = ["rgba(255,255,255,.9)", "#ffd23f", "#ff3b30"];
    for (const [x, y, role] of marks || []) {
      ctx.strokeStyle = style[role]; ctx.lineWidth = role === 0 ? 2 : 3;
      if (role === 0) ctx.setLineDash([4, 3]);
      ctx.strokeRect(x * T + 1.5, y * T + 1.5, T - 3, T - 3);
      ctx.setLineDash([]);
    }
  };
  b.shade = function (tiles, fill) {
    const T = b.T;
    ctx.fillStyle = fill;
    for (const [x, y] of tiles) ctx.fillRect(x * T, y * T, T, T);
  };
  b.outline = function (tiles, stroke, width = 2) {
    const T = b.T;
    ctx.strokeStyle = stroke; ctx.lineWidth = width;
    for (const [x, y] of tiles) ctx.strokeRect(x * T + width / 2, y * T + width / 2, T - width, T - width);
  };
  b.path = function (tiles, stroke) {
    if (!tiles || tiles.length < 2) return;
    const T = b.T;
    ctx.strokeStyle = stroke; ctx.lineWidth = Math.max(3, T * .18); ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath();
    tiles.forEach(([x, y], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x * T + T / 2, y * T + T / 2));
    ctx.stroke();
    const [x, y] = tiles[tiles.length - 1];
    ctx.fillStyle = stroke; ctx.beginPath(); ctx.arc(x * T + T / 2, y * T + T / 2, T * .16, 0, 7); ctx.fill();
  };
  // The board for a frame; skip: a unit id to leave out (one being animated);
  // under: drawn over the terrain, under the units (ranges).
  b.draw = function (f, skip, under) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) drawTile(x, y, f);
    if (under) under();
    for (const u of f.u) if (!(u[8] & CARRIED) && u[0] !== skip) b.drawUnit(u);
  };
  // A tile's terrain, capture points and units, as tooltip HTML.
  b.tileInfo = function (f, x, y) {
    const esc = b.escape, id = M.terrain[y][x];
    const info = M.terrain_kinds[id] || ["", "?"];
    let html = `<b>(${x},${y})</b> ${esc(info[1])}`;
    if (info[2] !== undefined) html += ` · ${"★".repeat(info[2]) || "no"} defence`;
    const p = propAt[x + "," + y];
    if (p !== undefined && f.o[p] < 2) html += ` · <span style="color:${color(f.o[p])}">■</span> P${f.o[p] + 1}`;
    if (p !== undefined && BUILDING[b.kindAt(x, y, f)] && f.c[p] < 20) html += ` · capture ${f.c[p]}/20`;
    if (p !== undefined && b.kindAt(x, y, f) === "seam") html += ` · ${f.c[p]} HP`;
    const here = f.u.filter((u) => u[3] === x && u[4] === y && !(u[8] & CARRIED));
    for (const u of here) {
      html += `<br><span style="color:${color(u[2])}">■</span> ${esc(G.units[u[1]])}: ${hpText(u[5])} HP, fuel ${u[6]}, ammo ${u[7]}`;
      const flags = [];
      if (u[8] & MOVED) flags.push("moved"); if (u[8] & HIDDEN) flags.push("hidden");
      if (u[8] & STUNNED) flags.push("stunned"); if (u[8] & CAPTURING) flags.push("capturing");
      if (flags.length) html += ` (${flags.join(", ")})`;
      for (const c of [u[9], u[10]]) {
        const cu = c && f.u.find((v) => v[0] === c);
        if (cu) html += `<br>&nbsp;&nbsp;carrying ${esc(G.units[cu[1]])} ${hpText(cu[5])} HP`;
      }
    }
    return html;
  };
  b.escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return b;
}
