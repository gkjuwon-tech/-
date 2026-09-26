// 바이킹 드워프 헬멧 — 점 찍고, 선 잇고, 면 만드는 수제 로우폴리 생성기
// 좌표계: +Y 위, +Z 앞(얼굴 방향), +X 캐릭터 기준 왼쪽. 단위 1 = 대가리 반지름.
// 출력: helmet.obj (파츠별 오브젝트, 쿼드 위주) + helmet.json (뷰어용)

const fs = require('fs');
const path = require('path');

// ---------- 벡터 유틸 ----------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => mul(a, 1 / (len(a) || 1));
const lerp = (a, b, t) => a + (b - a) * t;
const deg = (d) => (d * Math.PI) / 180;

// ---------- 메쉬 ----------
const verts = [];
const faces = []; // { idx:[], part }
const landmarks = []; // { name, p }
let part = 'dome';

function V(p) { verts.push(p); return verts.length - 1; }
function mark(name, p) { landmarks.push({ name, p: p.map((v) => +v.toFixed(4)) }); }

// ref: 면이 바깥을 보게 할 기준점 함수 (centroid -> 안쪽 점). inward=true면 안쪽을 보게.
function F(idx, ref, inward = false) {
  if (ref) {
    const ps = idx.map((i) => verts[i]);
    const c = mul(ps.reduce(add, [0, 0, 0]), 1 / ps.length);
    let n = [0, 0, 0];
    for (let i = 0; i < ps.length; i++) n = add(n, cross(ps[i], ps[(i + 1) % ps.length]));
    const out = sub(c, ref(c));
    if ((dot(n, out) < 0) !== inward) idx = idx.slice().reverse();
  }
  faces.push({ idx, part });
}

// 링 두 개를 쿼드 띠로 잇기
function strip(A, B, closed, ref, inward) {
  const n = A.length;
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const j = (i + 1) % n;
    F([A[i], A[j], B[j], B[i]], ref, inward);
  }
}
// 링을 한 점으로 모으는 부채꼴 (꼭대기, 뿔 끝)
function fan(ring, tip, ref) {
  for (let i = 0; i < ring.length; i++) F([ring[i], ring[(i + 1) % ring.length], tip], ref);
}

// =====================================================================
// 1) 돔 (투구 본체) — 달걀형 타원체, 뒤통수 살짝 더 큼
// =====================================================================
const RX = 1.0, RY = 1.08, RZ = 1.12;
const NA = 32; // 둘레 분할

function domePt(theta, a, s = 1) {
  // theta: 0=정면, +90°=+X 옆. a: 정수리에서 내려온 각도
  const back = Math.max(0, -Math.cos(theta));
  const rz = RZ * (1 + 0.05 * back);
  return [RX * s * Math.sin(a) * Math.sin(theta), RY * s * Math.cos(a), rz * s * Math.sin(a) * Math.cos(theta)];
}
// 테두리 높이: 정면은 눈썹 위, 옆은 귀 덮게 쭉 내려옴, 뒤는 목덜미
const aRim = (t) => deg(97) + deg(12) * Math.sin(t) ** 2 + deg(6) * Math.max(0, -Math.cos(t));
const aBand = (t) => aRim(t) - deg(17);
const origin = () => [0, 0, 0];
const thetas = [...Array(NA)].map((_, i) => (i / NA) * Math.PI * 2);

part = 'dome';
const top = V([0, RY, 0]);
mark('정수리', verts[top]);
const domeRings = [];
const NLAT = 11;
for (let k = 1; k <= NLAT; k++) {
  const u = k / NLAT;
  domeRings.push(thetas.map((t) => V(domePt(t, u * aBand(t)))));
}
fan(domeRings[0], top, origin);
for (let k = 0; k < domeRings.length - 1; k++) strip(domeRings[k], domeRings[k + 1], true, origin);

// =====================================================================
// 2) 이마 띠 (brow band) — 한 단 튀어나온 두꺼운 쇠 띠 + 안쪽 립
// =====================================================================
part = 'band';
const bandProfile = [
  // [각도 함수, 반지름 배율]
  [(t) => aBand(t), 1.0],
  [(t) => aBand(t), 1.055],
  [(t) => aBand(t) + deg(2.5), 1.075],
  [(t) => aRim(t) - deg(2.5), 1.075],
  [(t) => aRim(t), 1.05],
  [(t) => aRim(t) + deg(1), 0.96],
  [(t) => aRim(t) - deg(9), 0.95],
];
const bandRings = bandProfile.map(([af, s]) => thetas.map((t) => V(domePt(t, af(t), s))));
for (let k = 0; k < bandRings.length - 1; k++) strip(bandRings[k], bandRings[k + 1], true, origin, k >= 5);
mark('이마 띠 정면', domePt(0, aBand(0) + deg(8), 1.075));
mark('귀 덮개 끝', domePt(Math.PI / 2, aRim(Math.PI / 2), 1.05));
mark('목덜미', domePt(Math.PI, aRim(Math.PI), 1.05));

// 리벳 — 띠 따라 박힌 징
part = 'rivets';
function rivet(p, n, r) {
  const up = Math.abs(n[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = norm(cross(n, up)), v = cross(n, u);
  const ring = (rr, h, k) => [...Array(k)].map((_, i) => {
    const a = (i / k) * Math.PI * 2;
    return V(add(add(p, mul(n, h)), add(mul(u, Math.cos(a) * rr), mul(v, Math.sin(a) * rr))));
  });
  const r0 = ring(r, -0.005, 6), r1 = ring(r * 0.75, r * 0.55, 6);
  const tip = V(add(p, mul(n, r * 0.8)));
  const ref = () => p;
  strip(r0, r1, true, ref);
  fan(r1, tip, ref);
}
for (let i = 0; i < 22; i++) {
  const t = (i / 22) * Math.PI * 2 + Math.PI / 22;
  const a = (aBand(t) + aRim(t)) / 2;
  const p = domePt(t, a, 1.075);
  rivet(p, norm(sub(domePt(t, a, 1.2), p)), 0.038);
}

// =====================================================================
// 3) 볏 (crest) — 앞이마에서 뒤통수까지 톱니 달린 쇠 지느러미
// =====================================================================
part = 'crest';
const NS = 41;
const crestSecs = [];
for (let k = 0; k < NS; k++) {
  const s = k / (NS - 1); // 0=뒤, 1=앞
  const phi = lerp(-deg(78), deg(70), s); // YZ 평면에서 정수리 기준 각
  const theta = phi >= 0 ? 0 : Math.PI;
  const p = domePt(theta, Math.abs(phi), 0.99);
  const n = norm(sub(domePt(theta, Math.abs(phi), 1.1), p));
  const env = Math.sin(Math.PI * s) ** 0.6;
  const tooth = k % 4 === 0 ? 1 : k % 4 === 2 ? 0 : 0.5; // 톱니
  const h = env * (0.07 + 0.09 * tooth) + 0.005;
  const w = 0.012 + 0.045 * env;
  const X = [1, 0, 0];
  const sec = [
    add(p, mul(X, -w * 1.5)),
    add(add(p, mul(n, h * 0.45)), mul(X, -w)),
    add(p, mul(n, h)),
    add(add(p, mul(n, h * 0.45)), mul(X, w)),
    add(p, mul(X, w * 1.5)),
  ].map(V);
  crestSecs.push(sec);
  if (k === 0) mark('볏 뒤끝', verts[sec[2]]);
  if (k === NS - 1) mark('볏 앞끝', verts[sec[2]]);
  if (k === 20) mark('볏 꼭대기 톱니', verts[sec[2]]);
}
const crestRef = (c) => [0, c[1] * 0.9, c[2] * 0.9];
for (let k = 0; k < NS - 1; k++) strip(crestSecs[k], crestSecs[k + 1], false, crestRef);

// =====================================================================
// 4) 코가리개 (nose guard)
// =====================================================================
part = 'noseguard';
{
  const top0 = domePt(0, aBand(0), 1.05);
  const rimF = domePt(0, aRim(0), 1.05);
  const NG = 9;
  const secs = [];
  for (let k = 0; k < NG; k++) {
    const t = k / (NG - 1);
    const y = lerp(top0[1] + 0.02, rimF[1] - 0.42, t);
    const z = lerp(top0[2], rimF[2] + 0.02, Math.min(1, t * 3)) + 0.05 * t * t;
    const w = lerp(0.15, 0.065, t) * (k === NG - 1 ? 0.6 : 1);
    const th = 0.05 * (k === NG - 1 ? 0.7 : 1);
    const ridge = 0.022;
    const loop = [
      [-w, 0], [-w, th], [-w * 0.45, th + ridge * 0.6], [0, th + ridge],
      [w * 0.45, th + ridge * 0.6], [w, th], [w, 0],
    ].map(([x, dz]) => V([x, y, z - 0.02 + dz]));
    secs.push(loop);
  }
  const ref = (c) => [0, c[1], c[2] - 0.2];
  for (let k = 0; k < NG - 1; k++) strip(secs[k], secs[k + 1], true, ref);
  const bottom = secs[NG - 1];
  const tip = V(add(verts[bottom[3]], [0, -0.07, -0.01]));
  fan(bottom, tip, (c) => [c[0], c[1] + 0.3, c[2] - 0.05]);
  F(secs[0].slice(), (c) => [c[0], c[1] - 0.3, c[2]]);
  mark('코가리개 끝', verts[tip]);
}


// =====================================================================
// 4-2) 화난 눈썹 가드 — 코 쪽으로 푹 꺼지는 V자 쇠 눈썹 (드워프 빡침 지수 +200%)
// =====================================================================
part = 'brows';
for (const sx of [1, -1]) {
  const NB = 10;
  const secs = [];
  for (let k = 0; k < NB; k++) {
    const u = k / (NB - 1); // 0=코 쪽, 1=관자놀이
    const t = sx * lerp(deg(9), deg(58), u);
    const a = aRim(t) + deg(5) * (1 - u) ** 1.6 - deg(5) * u;
    const env = 0.55 + 0.45 * Math.sin(Math.PI * Math.min(1, u * 1.3));
    secs.push([
      [a - deg(3.2) * env, 1.07], [a - deg(1.2) * env, 1.13], [a + deg(2.4) * env, 1.115], [a + deg(3.2) * env, 1.04],
    ].map(([aa, s]) => V(domePt(t, aa, s))));
    if (k === 0) mark(sx > 0 ? '왼눈썹 미간' : '오른눈썹 미간', domePt(t, a, 1.13));
  }
  const ref = (c) => mul(c, 0.8);
  for (let k = 0; k < NB - 1; k++) strip(secs[k], secs[k + 1], true, ref);
  F(secs[0].slice(), (c) => domePt(0, aRim(0), 1.1));
  F(secs[NB - 1].slice(), (c) => domePt(sx * deg(20), aRim(0), 1.1));
}

// =====================================================================
// 5) 뿔 (horns) + 뿔 소켓 — 베지어 등뼈 따라 링을 쌓는다
// =====================================================================
function bez(P, t) {
  const m = 1 - t;
  return add(add(mul(P[0], m * m * m), mul(P[1], 3 * m * m * t)), add(mul(P[2], 3 * m * t * t), mul(P[3], t * t * t)));
}

function horn(side) {
  const sx = side; // +1 = +X, -1 = -X
  const base = domePt(sx * Math.PI / 2, deg(64), 0.86);
  const P = [
    base,
    [sx * 1.75, 0.62, 0.1],
    [sx * 2.25, 1.35, 0.05],
    [sx * 1.72, 2.3, -0.28],
  ];
  const NT = 26, NC = 12;
  // 평행 이동 프레임
  const pts = [], tans = [];
  for (let k = 0; k <= NT; k++) {
    const t = k / NT;
    pts.push(bez(P, t));
    tans.push(norm(sub(bez(P, Math.min(1, t + 1e-3)), bez(P, Math.max(0, t - 1e-3)))));
  }
  let N = norm(cross(tans[0], [0, 0, 1]));
  const frames = [];
  for (let k = 0; k <= NT; k++) {
    if (k > 0) {
      const b = cross(tans[k - 1], tans[k]);
      if (len(b) > 1e-6) {
        const ax = norm(b), ang = Math.acos(Math.min(1, dot(tans[k - 1], tans[k])));
        // 로드리게스 회전
        N = add(add(mul(N, Math.cos(ang)), mul(cross(ax, N), Math.sin(ang))), mul(ax, dot(ax, N) * (1 - Math.cos(ang))));
      }
    }
    frames.push([N, norm(cross(tans[k], N))]);
  }

  part = 'horns';
  const rings = [];
  for (let k = 0; k < NT; k++) {
    const t = k / NT;
    let r = 0.3 * Math.pow(1 - t, 0.75) + 0.02;
    if (k > 1 && k % 3 === 0) r *= 0.92; // 뿔 마디 홈
    const [n1, n2] = frames[k];
    const twist = t * 1.2;
    rings.push([...Array(NC)].map((_, i) => {
      const a = (i / NC) * Math.PI * 2 + twist;
      const e = 1 - 0.14 * Math.max(0, Math.cos(a - twist)); // 안쪽 면 살짝 납작
      return V(add(pts[k], add(mul(n1, Math.cos(a) * r * e), mul(n2, Math.sin(a) * r * 0.88))));
    }));
  }
  const tip = V(add(pts[NT], mul(tans[NT], 0.04)));
  const spineRef = (c) => {
    let best = pts[0], bd = 1e9;
    for (const q of pts) { const d = len(sub(c, q)); if (d < bd) { bd = d; best = q; } }
    return best;
  };
  for (let k = 0; k < NT - 1; k++) strip(rings[k], rings[k + 1], true, spineRef);
  fan(rings[NT - 1], tip, spineRef);
  const tag = sx > 0 ? '왼' : '오른';
  mark(`${tag}뿔 뿌리`, pts[2]);
  mark(`${tag}뿔 꺾임`, bez(P, 0.55));
  mark(`${tag}뿔 끝`, verts[tip]);

  // 소켓: 뿔 뿌리를 감싸는 쇠 고리 (선반 모양 회전체)
  part = 'sockets';
  const k0 = 3;
  const c0 = pts[k0], T = tans[k0], [n1, n2] = frames[k0];
  const prof = [[0.3, -0.12], [0.39, -0.1], [0.43, -0.04], [0.43, 0.04], [0.38, 0.09], [0.3, 0.1]];
  const NCs = 16;
  const srings = prof.map(([r, h]) => [...Array(NCs)].map((_, i) => {
    const a = (i / NCs) * Math.PI * 2;
    return V(add(add(c0, mul(T, h)), add(mul(n1, Math.cos(a) * r), mul(n2, Math.sin(a) * r))));
  }));
  const sref = (c) => add(c0, mul(T, dot(sub(c, c0), T)));
  for (let k = 0; k < srings.length - 1; k++) strip(srings[k], srings[k + 1], true, sref);
  part = 'rivets';
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 + 0.3;
    const d = add(mul(n1, Math.cos(a)), mul(n2, Math.sin(a)));
    rivet(add(c0, mul(d, 0.43)), d, 0.035);
  }
}
horn(+1);
horn(-1);

// =====================================================================
// 출력
// =====================================================================
const outDir = __dirname;
const parts = [...new Set(faces.map((f) => f.part))];
let obj = '# 바이킹 드워프 헬멧 — 수제 로우폴리\n# +Y up, +Z front\n';
for (const p of verts) obj += `v ${p.map((v) => v.toFixed(5)).join(' ')}\n`;
for (const pn of parts) {
  obj += `o Helmet_${pn}\n`;
  for (const f of faces) if (f.part === pn) obj += `f ${f.idx.map((i) => i + 1).join(' ')}\n`;
}
fs.writeFileSync(path.join(outDir, 'helmet.obj'), obj);

const tris = faces.reduce((s, f) => s + f.idx.length - 2, 0);
const edgeSet = new Set();
for (const f of faces) f.idx.forEach((a, i) => {
  const b = f.idx[(i + 1) % f.idx.length];
  edgeSet.add(a < b ? `${a}_${b}` : `${b}_${a}`);
});
const stats = { objects: parts.length, vertices: verts.length, edges: edgeSet.size, faces: faces.length, triangles: tris };
fs.writeFileSync(path.join(outDir, 'helmet.json'), JSON.stringify({
  stats, parts,
  verts: verts.flat().map((v) => +v.toFixed(4)),
  faces: faces.map((f) => [parts.indexOf(f.part), ...f.idx]),
  landmarks,
}));
console.log(stats, parts, `landmarks: ${landmarks.length}`);

// 뷰어에 데이터 박아넣기 (파일 하나로 폰에서도 열리게)
const tpl = fs.readFileSync(path.join(outDir, 'viewer.template.html'), 'utf8');
fs.writeFileSync(path.join(outDir, 'helmet.html'), tpl.replace('/*DATA*/null', fs.readFileSync(path.join(outDir, 'helmet.json'), 'utf8')));
