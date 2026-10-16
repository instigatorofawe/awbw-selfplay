// The board as the viewer and the play page draw it: terrain, buildings and
// units on a canvas, from a game's static data (G: map, units, players,
// countries) and a frame (viewer._frame: units, owners, capture points).
// With G.sprites (sprites.py: the Replay Player's art from a local copy) it
// draws the Player's sprites, animated once the page passes its redraw to
// animate(); without, or for anything the copy lacks, coloured squares.
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
  // top: CSS px above row 0 for the top row's tall sprites (the board starts there)
  const b = { T: 28, top: 0, W, H, MOVED, HIDDEN, CARRIED, STUNNED, CAPTURING, BUILDING, color, propAt, exactHp: true };
  const hpText = (hp) => (b.exactHp ? (hp / 10).toFixed(1) : String(Math.ceil(hp / 10)));

  // Sprites: S.img holds data URIs by key; refs are [[key per weather], timing]
  // and a timing (an index into S.anim, frame lengths in ms) adds "-<frame>".
  const S = G.sprites || null, img = {}, grey = {}, holes = {};
  let ready = false;
  const greyed = (im) => {  // the Player's greyscale for units that have acted, a little darker
    try {
      const c = document.createElement("canvas");
      c.width = im.width; c.height = im.height;
      const g = c.getContext("2d");
      g.drawImage(im, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height), a = d.data;
      for (let i = 0; i < a.length; i += 4) a[i] = a[i + 1] = a[i + 2] = ((a[i] + a[i + 1] + a[i + 2]) / 3) * 0.85;
      g.putImageData(d, 0, 0);
      return c;
    } catch (e) { return null; }
  };
  const seeThrough = (im) => {  // any see-through pixel in its tile's 16x16 (drawn over plain)
    try {
      const c = document.createElement("canvas");
      c.width = 16; c.height = 16;
      const g = c.getContext("2d");
      g.drawImage(im, (16 - im.width) / 2, 16 - im.height);
      const a = g.getImageData(0, 0, 16, 16).data;
      for (let i = 3; i < a.length; i += 4) if (a[i] < 255) return true;
      return false;
    } catch (e) { return false; }
  };
  if (S) {
    const keys = Object.keys(S.img);
    let left = keys.length;
    const done = () => { if (--left === 0) { ready = true; if (b.redraw) b.redraw(); } };
    for (const k of keys) {
      const im = new Image();
      im.onload = () => {
        img[k] = im;
        if (k.startsWith("Units/")) grey[k] = greyed(im);
        else if (k.startsWith("Map/")) holes[k] = seeThrough(im);
        done();
      };
      im.onerror = done;
      im.src = S.img[k];
    }
    if (!keys.length) ready = true;
  }
  const totals = S ? S.anim.map((d) => d.reduce((a, x) => a + x, 0)) : [];
  function frameAt(i, now) {
    const d = S.anim[i];
    let t = now % totals[i];
    for (let f = 0; f < d.length; f++) { if (t < d[f]) return f; t -= d[f]; }
    return 0;
  }
  function sprite(ref, w, now) {  // the loaded image a ref shows now, or null
    if (!ready || !ref) return null;
    const tail = ref[1] < 0 ? "" : "-" + frameAt(ref[1], now);
    const k = (ref[0][w] || ref[0][0]) + tail;
    return img[k] ? k : img[ref[0][0] + tail] ? ref[0][0] + tail : null;
  }
  function blit(im, px, py, flip, centre) {  // bottom-aligned in the tile, or centred
    const s = b.T / 16, w = im.width * s, h = im.height * s;
    const x = px + (b.T - w) / 2, y = centre ? py + (b.T - h) / 2 : py + b.T - h;
    if (!flip) { ctx.drawImage(im, x, y, w, h); return; }
    ctx.save(); ctx.translate(x + w, y); ctx.scale(-1, 1); ctx.drawImage(im, 0, 0, w, h); ctx.restore();
  }
  // Redraws the board with redraw() whenever an animation moves on a frame.
  b.animate = function (redraw) {
    b.redraw = redraw;
    if (!S) return;
    let last = "";
    const tick = () => {
      const now = performance.now();
      if (ready) {
        const sig = totals.map((_, i) => frameAt(i, now)).join() + "," + Math.floor(now / 1000);
        if (sig !== last) { last = sig; redraw(); }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  b.size = function (avail, max = 36) {
    let T = Math.max(14, Math.min(max, Math.floor(avail / W)));
    const dpr = window.devicePixelRatio || 1;
    if (S) {  // whole device pixels to a sprite pixel, unless that shrinks the board much
      const snapped = (Math.floor((T * dpr) / 16) * 16) / dpr;
      if (snapped >= 0.8 * T) T = snapped;
    }
    b.T = T;
    b.top = S ? (S.pad * T) / 16 : 0;
    cv.width = Math.round(W * T * dpr); cv.height = Math.round((H * T + b.top) * dpr);
    cv.style.width = W * T + "px"; cv.style.height = H * T + b.top + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, b.top * dpr);
    ctx.imageSmoothingEnabled = false;
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
    const x = Math.floor((e.clientX - r.left) / b.T), y = Math.floor((e.clientY - r.top - b.top) / b.T);
    return x < 0 || y < 0 || x >= W || y >= H ? null : [x, y];
  };
  function captureBar(px, py, points) {
    const T = b.T;
    ctx.fillStyle = "#000"; ctx.fillRect(px + 2, py + T - 5, T - 4, 3);
    ctx.fillStyle = "#ffd23f"; ctx.fillRect(px + 2, py + T - 5, (T - 4) * points / 20, 3);
  }
  function drawTileSprite(x, y, f, now) {
    const w = f.w || 0, px = x * b.T, py = y * b.T;
    const g = sprite(S.ground[y * W + x], w, now);
    if (!g) return false;
    const p = propAt[x + "," + y], P = p === undefined ? null : S.props[p];
    let top = null;
    if (P) {
      top = sprite(P.x && f.c[p] === 0 ? P.x : P[f.o[p]] || P.n, w, now);
      if (!top) return false;
    }
    const under = holes[g] && sprite(S.plain, w, now);
    if (under) blit(img[under], px, py);
    blit(img[g], px, py);
    if (top) blit(img[top], px, py);
    if (P && BUILDING[b.kindAt(x, y, f)] && f.c[p] < 20) captureBar(px, py, f.c[p]);
    return true;
  }
  function drawTile(x, y, f, now) {
    if (S && drawTileSprite(x, y, f, now)) return;
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
      if (p !== undefined && f.c[p] < 20) captureBar(px, py, f.c[p]);  // being captured
    }
    ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1;
    ctx.strokeRect(px + .5, py + .5, T - 1, T - 1);
  }
  // A unit as a sprite: greyed once it has acted, its HP and one status icon
  // at a time (capturing, cargo, hidden, stunned, low fuel, low ammo).
  function drawUnitSprite(u, at, dim, dir) {
    const [, type, owner, ux, uy, hp, fuel, ammo, flags] = u;
    const now = performance.now(), T = b.T, s = T / 16;
    const x = at ? at[0] : ux, y = at ? at[1] : uy, px = x * T, py = y * T;
    // idle sprites face the country's way; side moves face east, flipped going west
    let ref = S.units[owner] && S.units[owner][type], flip = false, moving = false;
    const m = dir && S.moves && S.moves[owner] && S.moves[owner][type];
    const mref = m && m[dir === "N" ? "U" : dir === "S" ? "D" : "S"];
    if (mref) { ref = mref; moving = true; flip = dir === "W"; }
    const k = sprite(ref, 0, now);
    if (!k) return false;
    blit(((dim ?? flags & MOVED) && grey[k]) || img[k], px, py, flip, moving);
    const icon = (key) => { const im = key && img[key]; return im ? [im, im.width * s, im.height * s] : null; };
    const shown = Math.ceil(hp / 10), d = shown < 10 && icon(S.ui.hp[shown]);
    if (d) ctx.drawImage(d[0], px + T - d[1], py + T - d[2], d[1], d[2]);
    const low = G.unit_stats && G.unit_stats[type], icons = [];
    if (flags & CAPTURING) icons.push("capture");
    if (u[9] || u[10]) icons.push("cargo");
    if (flags & HIDDEN) icons.push("dive");
    if (flags & STUNNED) icons.push("stun");
    if (low && fuel * 3 <= low[0]) icons.push("fuel");  // a third or less left
    if (low && low[1] > 0 && ammo * 3 <= low[1]) icons.push("ammo");
    const c = icons.length && icon(S.ui[icons[Math.floor(now / 1000) % icons.length]]);
    if (c) ctx.drawImage(c[0], px, py + T - c[2], c[1], c[2]);
    return true;
  }
  // A unit row (viewer._frame), drawn at (x, y) if given (an animated move,
  // heading dir: N, E, S or W); dim: greyed, by default once it has acted.
  b.drawUnit = function (u, at, dim, dir) {
    if (S && drawUnitSprite(u, at, dim, dir)) return;
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
  // The Player's arrow: a body (N-S), a curve (N to W) and a tip (from N,
  // pointing S), turned to fit; each tile after the unit's.
  const TURN = { N: 2, E: 3, S: 0, W: 1 };  // quarter turns for a tip heading that way
  const CURVE = { NW: 0, NE: 1, ES: 2, SW: 3 };  // for a curve joining those sides
  const heading = (a, c) => (c[0] > a[0] ? "E" : c[0] < a[0] ? "W" : c[1] < a[1] ? "N" : "S");
  const back = { N: "S", S: "N", E: "W", W: "E" };
  function arrow(tiles) {
    const body = img[S.ui.body], curve = img[S.ui.curve], tip = img[S.ui.tip];
    if (!ready || !body || !curve || !tip) return false;
    const T = b.T;
    for (let i = 1; i < tiles.length; i++) {
      const [x, y] = tiles[i], from = back[heading(tiles[i - 1], tiles[i])];
      let im = tip, turns = TURN[heading(tiles[i - 1], tiles[i])];
      if (i + 1 < tiles.length) {
        const to = heading(tiles[i], tiles[i + 1]);
        if (to === back[from]) { im = body; turns = to === "N" || to === "S" ? 0 : 1; }
        else { im = curve; turns = CURVE[[from, to].sort((p, q) => "NESW".indexOf(p) - "NESW".indexOf(q)).join("")]; }
      }
      ctx.save();
      ctx.translate(x * T + T / 2, y * T + T / 2);
      ctx.rotate((turns * Math.PI) / 2);
      ctx.drawImage(im, -T / 2, -T / 2, T, T);
      ctx.restore();
    }
    return true;
  }
  b.path = function (tiles, stroke) {
    if (!tiles || tiles.length < 2) return;
    if (S && arrow(tiles)) return;
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
    const now = performance.now();
    ctx.clearRect(0, -b.top, W * b.T, H * b.T + b.top);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) drawTile(x, y, f, now);
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
