// 사진 역투영 피팅기: trace.json의 2D 점들에 맞도록
// 카메라(약원근) + 뿔 베지어 3D 제어점 + 뿔 두께 + 돔 크기를 넬더-미드로 동시 최적화.
// 좌우 뿔 대칭 제약이 깊이를 결정해준다.
const fs = require('fs');
const path = require('path');
const T = JSON.parse(fs.readFileSync(path.join(__dirname, 'trace.json'), 'utf8'));

const deg = (d) => (d * Math.PI) / 180;
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

const DOME_RX = 1.0, DOME_RZ = 1.1;

function unpack(x) {
  return {
    yaw: x[0], pitch: x[1], roll: x[2], s: x[3], tx: x[4], ty: x[5],
    P: [[x[6], x[7], x[8]], [x[9], x[10], x[11]], [x[12], x[13], x[14]], [x[15], x[16], x[17]]],
    r0: x[18], rExp: x[19], H: x[20], rimA: x[21],
  };
}
function projector(c) {
  const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
  const cr = Math.cos(c.roll), sr = Math.sin(c.roll);
  return (p) => {
    // Ry(yaw) → Rx(pitch) → 화면 roll
    const x1 = cy * p[0] + sy * p[2], z1 = -sy * p[0] + cy * p[2];
    const y2 = cp * p[1] - sp * z1;
    const u = x1 * cr - y2 * sr, v = x1 * sr + y2 * cr;
    return [c.tx + c.s * u, c.ty - c.s * v];
  };
}
const bez = (P, t) => {
  const m = 1 - t;
  return add(add(mul(P[0], m * m * m), mul(P[1], 3 * m * m * t)), add(mul(P[2], 3 * m * t * t), mul(P[3], t * t * t)));
};
const mirror = (P) => P.map((p) => [-p[0], p[1], p[2]]);
const domePt = (c, theta, a) => [DOME_RX * Math.sin(a) * Math.sin(theta), c.H * Math.cos(a), DOME_RZ * Math.sin(a) * Math.cos(theta)];

const NSAMP = 80;
function hornLoss(c, pr, P, obs, report) {
  const samples = [];
  for (let i = 0; i <= NSAMP; i++) samples.push(pr(bez(P, i / NSAMP)));
  let L = 0, lastT = -1;
  const out = [];
  obs.forEach((o, k) => {
    const isTip = k === obs.length - 1;
    let bi = NSAMP, bd = Infinity;
    if (isTip) bd = (samples[NSAMP][0] - o.p[0]) ** 2 + (samples[NSAMP][1] - o.p[1]) ** 2;
    else for (let i = 0; i <= NSAMP; i++) {
      const d = (samples[i][0] - o.p[0]) ** 2 + (samples[i][1] - o.p[1]) ** 2;
      if (d < bd) { bd = d; bi = i; }
    }
    const t = bi / NSAMP;
    const wModel = 2 * c.s * c.r0 * Math.pow(1 - t, c.rExp);
    L += bd * (isTip ? 2 : 1) + 0.4 * (wModel - o.w) ** 2;
    if (t < lastT) L += 400 * (lastT - t) ** 2 * 100;
    lastT = t;
    out.push({ n: o.n, t: +t.toFixed(3), err: +Math.sqrt(bd).toFixed(2), wModel: +wModel.toFixed(1), w: o.w });
  });
  // 첫 관측점은 뿔이 헬멧 밖으로 나오는 초반부여야 함
  if (lastT < 0.95) L += 50;
  // 뿔이 돔을 관통하면 벌점
  for (let i = 8; i <= NSAMP; i += 4) {
    const p = bez(P, i / NSAMP);
    const q = (p[0] / DOME_RX) ** 2 + (p[1] / c.H) ** 2 + (p[2] / DOME_RZ) ** 2;
    if (q < 1.1 && p[1] > -0.3) L += 200 * (1.1 - q);
  }
  if (report) report.push(...out);
  return L;
}

function loss(x, report) {
  const c = unpack(x);
  const pr = projector(c);
  let L = 0;
  L += hornLoss(c, pr, c.P, T.nearHorn, report?.near);
  L += hornLoss(c, pr, mirror(c.P), T.farHorn, report?.far);

  // 돔 실루엣
  let minU = Infinity, minV = Infinity, topU = 0;
  for (let i = 0; i < 48; i++) for (let j = 0; j <= 12; j++) {
    const q = pr(domePt(c, (i / 48) * Math.PI * 2, (j / 12) * c.rimA));
    if (q[0] < minU) minU = q[0];
    if (q[1] < minV) { minV = q[1]; topU = q[0]; }
  }
  const fr = pr(domePt(c, 0, c.rimA));
  L += 2 * (minV - T.dome.top[1]) ** 2 + 0.3 * (topU - T.dome.top[0]) ** 2;
  L += 1 * (minU - T.dome.leftSilhouette[0]) ** 2;
  L += 0.6 * ((fr[0] - T.dome.frontRim[0]) ** 2 + (fr[1] - T.dome.frontRim[1]) ** 2);
  if (report) report.dome = { top: [topU, minV].map((v) => +v.toFixed(1)), left: +minU.toFixed(1), frontRim: fr.map((v) => +v.toFixed(1)) };

  // 정규화: 뿔 뿌리는 돔 옆면 약간 안쪽, 물리적으로 말 되는 값들
  const p0 = c.P[0];
  const q0 = Math.sqrt((p0[0] / DOME_RX) ** 2 + (p0[1] / c.H) ** 2 + (p0[2] / DOME_RZ) ** 2);
  L += 300 * (q0 - 0.85) ** 2 + 100 * Math.max(0, 0.6 - p0[0]) ** 2;
  L += 50 * Math.max(0, 0.15 - c.r0) ** 2 * 100 + 50 * Math.max(0, c.r0 - 0.7) ** 2 * 100;
  L += 20 * Math.max(0, 0.4 - c.rExp) ** 2 + 20 * Math.max(0, c.rExp - 2) ** 2;
  L += 50 * Math.max(0, 0.45 - c.H) ** 2 + 50 * Math.max(0, c.H - 1.3) ** 2;
  L += 200 * Math.max(0, Math.abs(c.roll) - deg(20)) ** 2;
  L += 200 * Math.max(0, deg(80) - c.rimA) ** 2 + 200 * Math.max(0, c.rimA - deg(110)) ** 2;
  // 제어점 폭주 방지
  for (const p of c.P) L += 0.05 * (p[0] ** 2 + p[1] ** 2 + p[2] ** 2);
  return L;
}

function nelderMead(f, x0, step, iters) {
  const n = x0.length;
  let S = [x0.slice()];
  for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += step[i]; S.push(x); }
  let F = S.map(f);
  for (let it = 0; it < iters; it++) {
    const o = [...Array(n + 1).keys()].sort((a, b) => F[a] - F[b]);
    S = o.map((i) => S[i]); F = o.map((i) => F[i]);
    const cen = Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cen[j] += S[i][j] / n;
    const pt = (a) => cen.map((c, j) => c + a * (S[n][j] - c));
    const xr = pt(-1), fr = f(xr);
    if (fr < F[0]) { const xe = pt(-2), fe = f(xe); if (fe < fr) { S[n] = xe; F[n] = fe; } else { S[n] = xr; F[n] = fr; } }
    else if (fr < F[n - 1]) { S[n] = xr; F[n] = fr; }
    else {
      const xc = pt(0.5), fc = f(xc);
      if (fc < F[n]) { S[n] = xc; F[n] = fc; }
      else for (let i = 1; i <= n; i++) { S[i] = S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j])); F[i] = f(S[i]); }
    }
  }
  const b = F.indexOf(Math.min(...F));
  return { x: S[b], f: F[b] };
}

// 시드 고정 난수
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

const base = [deg(35), deg(18), 0, 32, 628, 978,
  0.85, 0.25, 0.0, 1.9, 0.45, 0.35, 2.5, 1.3, 0.25, 2.1, 2.2, 0.0,
  0.42, 0.9, 0.75, deg(95)];
const step = [0.3, 0.2, 0.1, 6, 10, 10, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.3, 0.2, 0.2];

let best = null;
for (let r = 0; r < 60; r++) {
  const x0 = base.map((v, i) => v + (rnd() - 0.5) * step[i] * (r ? 2 : 0));
  if (r % 2) x0[0] = -x0[0]; // yaw 부호 둘 다 시도
  let res = nelderMead(loss, x0, step, 6000);
  for (let k = 0; k < 4; k++) res = nelderMead(loss, res.x, step.map((s) => s * 0.3), 4000);
  if (!best || res.f < best.f) { best = res; console.log(`restart ${r}: loss ${res.f.toFixed(2)}`); }
}

const report = { near: [], far: [] };
loss(best.x, report);
const c = unpack(best.x);
const result = {
  loss: +best.f.toFixed(3),
  camera: { yawDeg: +(c.yaw * 180 / Math.PI).toFixed(2), pitchDeg: +(c.pitch * 180 / Math.PI).toFixed(2), rollDeg: +(c.roll * 180 / Math.PI).toFixed(2), s: c.s, tx: c.tx, ty: c.ty },
  horn: { P: c.P.map((p) => p.map((v) => +v.toFixed(4))), r0: +c.r0.toFixed(4), rExp: +c.rExp.toFixed(4) },
  dome: { RX: DOME_RX, RZ: DOME_RZ, H: +c.H.toFixed(4), rimADeg: +(c.rimA * 180 / Math.PI).toFixed(2) },
  report,
};
fs.writeFileSync(path.join(__dirname, 'fit.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ camera: result.camera, horn: result.horn, dome: result.dome }, null, 1));
console.table(report.near); console.table(report.far); console.log(report.dome);
