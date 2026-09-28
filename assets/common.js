/* ============================================================
   MiniMind 从零学习项目 · 通用脚本
   提供：SVG 创建助手、简单动画播放器、侧边栏高亮
   ============================================================ */

/** 创建 SVG 元素（命名空间助手） */
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

/** 简单的帧动画播放器：把一组帧函数按 fps 依次调用 */
class Player {
  constructor(fps = 2) {
    this.fps = fps;
    this.frames = [];
    this.i = 0;
    this.timer = null;
    this.playing = false;
    this.onFrame = null;
  }
  load(frames) { this.frames = frames; this.i = 0; this.render(); return this; }
  render() { this.onFrame && this.onFrame(this.frames[this.i], this.i, this.frames.length); }
  step(d = 1) { this.i = (this.i + d + this.frames.length) % this.frames.length; this.render(); }
  play() {
    if (this.playing) return;
    this.playing = true;
    this.timer = setInterval(() => this.step(1), 1000 / this.fps);
    this.updateBtn && this.updateBtn();
  }
  pause() {
    this.playing = false;
    clearInterval(this.timer); this.timer = null;
    this.updateBtn && this.updateBtn();
  }
  toggle() { this.playing ? this.pause() : this.play(); }
  bind(btn) {
    this.updateBtn = () => { if (btn) btn.textContent = this.playing ? "⏸ 暂停" : "▶ 播放"; };
    btn && btn.addEventListener("click", () => this.toggle());
    this.updateBtn();
  }
}

/** 便捷：给某个动画绑定 播放/上一步/下一步 按钮 */
function wireControls(player, ids) {
  const play = document.getElementById(ids.play);
  const prev = document.getElementById(ids.prev);
  const next = document.getElementById(ids.next);
  player.bind(play);
  prev && prev.addEventListener("click", () => { player.pause(); player.step(-1); });
  next && next.addEventListener("click", () => { player.pause(); player.step(1); });
}

/** 数字保留 n 位小数 */
const fx = (v, n = 3) => Number(v).toFixed(n);

/** 侧边栏当前章节高亮 */
document.addEventListener("DOMContentLoaded", () => {
  const path = location.pathname.split("/").pop();
  document.querySelectorAll(".side a").forEach(a => {
    if (a.getAttribute("href") === path) a.classList.add("active");
  });
});
