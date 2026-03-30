/**
 * 2D demo: polytope C, point x0, unit normal μ = (cos θ, sin θ).
 * Cyan: shadow of C on the μ-line [min μ·x, max μ·x] μ.
 * Purple: on the μ-line, from polytope's outer extent to μ·x0 μ when x0 sticks past the halfspace.
 * Green: Euclidean projection x* = Π_C(x0). Dashed: segment x0 → x*.
 */
(function () {
  var canvas = document.getElementById("dual-proj-canvas");
  if (!canvas || !canvas.getContext) return;

  var ctx = canvas.getContext("2d");
  var slider = document.getElementById("dual-proj-angle");
  var snapBtn = document.getElementById("dual-proj-snap");
  var readout = document.querySelector(".dual-proj-readout");

  var W = 8;
  var H = 7;
  var pad = 48;
  var cw = canvas.width;
  var ch = canvas.height;
  var scale = Math.min((cw - 2 * pad) / W, (ch - 2 * pad) / H);
  var padX = pad + ((cw - 2 * pad) - W * scale) / 2;
  var padY = pad + ((ch - 2 * pad) - H * scale) / 2;

  function toCX(wx, wy) {
    return { x: padX + wx * scale, y: ch - padY - wy * scale };
  }

  function drawSegWorld(x0, y0, x1, y1, color, width) {
    var a = toCX(x0, y0);
    var b = toCX(x1, y1);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 3;
    ctx.lineCap = "round";
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function drawSegWorldDashed(x0, y0, x1, y1, color, width) {
    var a = toCX(x0, y0);
    var b = toCX(x1, y1);
    ctx.beginPath();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2;
    ctx.lineCap = "round";
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawPoly(vertices, fill, stroke) {
    if (!vertices.length) return;
    ctx.beginPath();
    var p = toCX(vertices[0][0], vertices[0][1]);
    ctx.moveTo(p.x, p.y);
    for (var i = 1; i < vertices.length; i++) {
      p = toCX(vertices[i][0], vertices[i][1]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.strokeStyle = stroke || "#eaeaea";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawDotWorld(wx, wy, r, fill, stroke) {
    var p = toCX(wx, wy);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  /** Diamond in world units (radius ~0.14) */
  function drawDiamondWorld(wx, wy, rw, fill, stroke) {
    var pr = rw * scale;
    var p = toCX(wx, wy);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - pr);
    ctx.lineTo(p.x + pr, p.y);
    ctx.lineTo(p.x, p.y + pr);
    ctx.lineTo(p.x - pr, p.y);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function drawAxes() {
    ctx.strokeStyle = "rgba(234,234,234,0.25)";
    ctx.lineWidth = 1;
    var o = toCX(0, 0);
    var xr = toCX(W, 0);
    var yu = toCX(0, H);
    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(xr.x, xr.y);
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(yu.x, yu.y);
    ctx.stroke();
  }

  var verts = [
    [2, 2],
    [5.2, 2.4],
    [3.6, 5.1],
  ];
  var x0 = [6.6, 3.4];

  function distSq(a, b) {
    var dx = a[0] - b[0];
    var dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  function pointInTriangle(p, a, b, c) {
    function sign(p1, p2, p3) {
      return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
    }
    var d1 = sign(p, a, b);
    var d2 = sign(p, b, c);
    var d3 = sign(p, c, a);
    var hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
    var hasPos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNeg && hasPos);
  }

  function closestOnSegment(p, a, b) {
    var ab = [b[0] - a[0], b[1] - a[1]];
    var ap = [p[0] - a[0], p[1] - a[1]];
    var ab2 = ab[0] * ab[0] + ab[1] * ab[1];
    if (ab2 < 1e-14) return [a[0], a[1]];
    var t = Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / ab2));
    return [a[0] + t * ab[0], a[1] + t * ab[1]];
  }

  function closestPointOnTriangle(p, a, b, c) {
    if (pointInTriangle(p, a, b, c)) return [p[0], p[1]];
    var c0 = closestOnSegment(p, a, b);
    var c1 = closestOnSegment(p, b, c);
    var c2 = closestOnSegment(p, c, a);
    var candidates = [c0, c1, c2];
    var best = candidates[0];
    var bestD = distSq(p, best);
    for (var i = 1; i < 3; i++) {
      var d = distSq(p, candidates[i]);
      if (d < bestD) {
        bestD = d;
        best = candidates[i];
      }
    }
    return best;
  }

  var v0 = verts[0];
  var v1 = verts[1];
  var v2 = verts[2];
  var xStar = closestPointOnTriangle(x0, v0, v1, v2);
  var distPrimal = Math.sqrt(distSq(x0, xStar));
  var dxOpt = x0[0] - xStar[0];
  var dyOpt = x0[1] - xStar[1];
  var thetaOptDeg =
    distPrimal < 1e-10 ? NaN : (Math.atan2(dyOpt, dxOpt) * 180) / Math.PI;
  var thetaOptRad =
    distPrimal < 1e-10 ? NaN : ((Math.atan2(dyOpt, dxOpt) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);

  function dot2(a, b) {
    return a[0] * b[0] + a[1] * b[1];
  }

  function scale2(k, v) {
    return [k * v[0], k * v[1]];
  }

  var DECIMAL_PLACES = 3;

  function fmt(x) {
    return Number(x).toFixed(DECIMAL_PLACES);
  }

  function render() {
    var t = slider ? parseFloat(slider.value) / 100 : 1.0;
    var theta = t;
    var mu = [Math.cos(theta), Math.sin(theta)];

    var projVerts = verts.map(function (v) {
      return dot2(mu, v);
    });
    var minT = Math.min.apply(null, projVerts);
    var maxT = Math.max.apply(null, projVerts);
    var t0 = dot2(mu, x0);

    ctx.clearRect(0, 0, cw, ch);

    drawAxes();

    var L = Math.max(W, H) * 2;
    drawSegWorld(
      -L * mu[0],
      -L * mu[1],
      L * mu[0],
      L * mu[1],
      "rgba(234,234,234,0.12)",
      1
    );

    var pMin = scale2(minT, mu);
    var pMax = scale2(maxT, mu);
    var p0 = scale2(t0, mu);

    drawSegWorld(pMin[0], pMin[1], pMax[0], pMax[1], "rgba(129,211,249,0.95)", 7);

    if (t0 > maxT + 1e-9) {
      drawSegWorld(pMax[0], pMax[1], p0[0], p0[1], "rgba(180,130,255,0.95)", 7);
    } else if (t0 < minT - 1e-9) {
      drawSegWorld(p0[0], p0[1], pMin[0], pMin[1], "rgba(180,130,255,0.95)", 7);
    }

    drawPoly(verts, "rgba(0,173,181,0.12)", "#00adb5");

    // Primal: x0 → x*
    drawSegWorldDashed(x0[0], x0[1], xStar[0], xStar[1], "rgba(150,220,130,0.85)", 2);
    drawDiamondWorld(xStar[0], xStar[1], 0.14, "rgba(120,200,100,0.95)", "#b8e0a0");

    var muLen = 1.2;
    var muTip = scale2(muLen, mu);
    drawSegWorld(0, 0, muTip[0], muTip[1], "rgba(234,234,234,0.7)", 2);
    drawDotWorld(muTip[0], muTip[1], 4, "rgba(234,234,234,0.9)");

    drawDotWorld(x0[0], x0[1], 7, "#eaeaea");

    drawDotWorld(pMin[0], pMin[1], 5, "#ff6b6b");
    drawDotWorld(pMax[0], pMax[1], 5, "#ff6b6b");
    drawDotWorld(p0[0], p0[1], 6, "transparent", "#ff6b6b");

    // Signed scalar along μ vs the shadow interval [minT, maxT]: 0 inside; >0 past max; <0 before min.
    var signedGapMu = 0;
    if (t0 > maxT + 1e-12) signedGapMu = t0 - maxT;
    else if (t0 < minT - 1e-12) signedGapMu = t0 - minT;

    var thetaDeg = (theta * 180) / Math.PI;

    if (readout) {
      readout.innerHTML =
        '<dl class="dual-proj-stats dual-proj-stats--compact">' +
        "<dt>Optimal distance</dt><dd>" +
        fmt(distPrimal) +
        "</dd>" +
        "<dt>Optimal angle</dt><dd>" +
        (isNaN(thetaOptDeg) ? "—" : fmt(thetaOptDeg) + "°") +
        "</dd>" +
        "<dt>Gap along μ</dt><dd>" +
        fmt(signedGapMu) +
        "</dd>" +
        "<dt>Current angle</dt><dd>" +
        fmt(thetaDeg) +
        "°</dd>" +
        "</dl>";
    }
  }

  if (slider) {
    slider.addEventListener("input", render);
  }
  if (slider && snapBtn) {
    snapBtn.addEventListener("click", function () {
      if (isNaN(thetaOptRad)) return;
      slider.value = String(thetaOptRad * 100);
      render();
    });
  }
  render();
})();
