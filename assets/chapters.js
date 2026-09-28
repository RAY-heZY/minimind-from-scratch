/* ============================================================
   章节清单（HTML 页面与 Notebook 一一对应）
   修改这里即可同步所有页面的侧边栏与翻页导航
   ============================================================ */

/* GitHub 仓库坐标 —— 由 tools/set_repo.py 自动写入，也可手动改这两行 */
const REPO = "RAY-heZY/minimind-from-scratch";
const BRANCH = "main";
const REPO_URL = `https://github.com/${REPO}`;
const nbUrl = (nb) => `${REPO_URL}/blob/${BRANCH}/notebooks/${nb}`;

const CHAPTERS = [
  { n: "01", file: "01_overview.html",  nb: "01_overview.ipynb",  part: "base",  title: "项目总览与 uv 环境",       desc: "MiniMind 是什么、整体架构、用 uv 搭好可复现的实验环境。" },
  { n: "02", file: "02_tokenizer.html", nb: "02_tokenizer.ipynb", part: "base",  title: "分词器与数据流水线",        desc: "BPE 分词、特殊 token、Dataset / DataLoader 的构造。" },
  { n: "03", file: "03_rmsnorm.html",   nb: "03_rmsnorm.ipynb",   part: "model", title: "RMSNorm 均方根归一化",      desc: "为什么用 RMSNorm 替代 LayerNorm，以及它的数值稳定性。" },
  { n: "04", file: "04_rope.html",      nb: "04_rope.ipynb",      part: "model", title: "RoPE 旋转位置编码",         desc: "用复数旋转把位置信息注入 Q/K，并支持 YaRN 外推。" },
  { n: "05", file: "05_attention.html", nb: "05_attention.ipynb", part: "model", title: "因果自注意力与 GQA / KV Cache", desc: "缩放点积注意力、因果掩码、分组查询注意力与增量解码。" },
  { n: "06", file: "06_swiglu.html",    nb: "06_swiglu.ipynb",    part: "model", title: "SwiGLU 前馈网络",           desc: "门控前馈层：gate / up / down 三投影与 SiLU 门控。" },
  { n: "07", file: "07_moe.html",       nb: "07_moe.ipynb",       part: "model", title: "MoE 混合专家",             desc: "路由、Top-K 专家选择、负载均衡辅助损失。" },
  { n: "08", file: "08_block_model.html", nb: "08_block_model.ipynb", part: "model", title: "Transformer Block 与模型组装", desc: "Pre-Norm 残差块、权重共享、logits 与交叉熵损失。" },
  { n: "09", file: "09_pretrain.html",  nb: "09_pretrain.ipynb",  part: "train", title: "预训练 Pretrain",           desc: "自回归语言建模、学习率调度、梯度累积与混合精度。" },
  { n: "10", file: "10_sft.html",       nb: "10_sft.ipynb",       part: "train", title: "指令微调 SFT",             desc: "对话模板、loss mask 只算回答部分、多轮数据组织。" },
  { n: "11", file: "11_lora.html",      nb: "11_lora.ipynb",      part: "train", title: "LoRA 低秩适配",            desc: "冻结底座、只训练低秩增量，以及权重合并。" },
  { n: "12", file: "12_distill.html",   nb: "12_distill.ipynb",   part: "train", title: "知识蒸馏",                 desc: "黑盒硬标签蒸馏与白盒 KL + 温度软标签蒸馏。" },
  { n: "13", file: "13_dpo.html",       nb: "13_dpo.ipynb",       part: "train", title: "DPO 直接偏好优化",         desc: "用 chosen / rejected 对的 log-odds 比值做偏好对齐。" },
  { n: "14", file: "14_rl.html",        nb: "14_rl.ipynb",        part: "train", title: "强化学习 GRPO / PPO / CISPO", desc: "策略项 × 优势项 × 正则项的统一视角与实现。" },
  { n: "15", file: "15_agent_rl.html",  nb: "15_agent_rl.ipynb",  part: "train", title: "Agentic RL 智能体强化学习", desc: "多轮工具调用轨迹的奖励设计与策略更新。" },
];

const PART_LABEL = { base: "基础", model: "模型", train: "训练" };

function renderSidebar(current) {
  const host = document.querySelector(".side");
  if (!host) return;
  const groups = [["base", "第一部分 · 基础"], ["model", "第二部分 · 模型"], ["train", "第三部分 · 训练"]];
  let html = "";
  for (const [key, label] of groups) {
    html += `<h4>${label}</h4>`;
    for (const c of CHAPTERS.filter(x => x.part === key)) {
      html += `<a href="${c.file}"><span class="n">${c.n}</span>${c.title}</a>`;
    }
  }
  const cur = CHAPTERS.find(x => x.n === current);
  html += `<h4>资源</h4>`;
  if (cur) {
    html += `<a href="${nbUrl(cur.nb)}" target="_blank" rel="noopener"><span class="n">📓</span>本章 Notebook ↗</a>`;
  }
  html += `<a href="${REPO_URL}" target="_blank" rel="noopener"><span class="n">⌘</span>GitHub 仓库 ↗</a>`;
  html += `<a href="index.html"><span class="n">☰</span>课程主页</a>`;
  host.innerHTML = html;
}

function renderPager(current) {
  const host = document.querySelector(".pager");
  if (!host) return;
  const i = CHAPTERS.findIndex(c => c.n === current);
  const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
  let html = "";
  html += prev
    ? `<a class="prev" href="${prev.file}"><span class="lbl">← 上一章</span><span class="ttl">${prev.n} ${prev.title}</span></a>`
    : `<a class="prev" href="index.html"><span class="lbl">← 返回</span><span class="ttl">课程主页</span></a>`;
  html += next
    ? `<a class="next" href="${next.file}"><span class="lbl">下一章 →</span><span class="ttl">${next.n} ${next.title}</span></a>`
    : `<a class="next" href="index.html"><span class="lbl">完成 →</span><span class="ttl">返回课程主页</span></a>`;
  host.innerHTML = html;
}

function bootChapter(current) {
  renderSidebar(current);
  renderPager(current);
}
