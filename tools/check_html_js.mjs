import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 站点根目录 = 仓库根目录；相对本脚本定位，无论从哪个工作目录运行都能找到
const here = path.dirname(fileURLToPath(import.meta.url));
const dir = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(here, '..');
const files = fs.readdirSync(dir).filter(f => f.endsWith('.html'));
let bad = 0;
for (const f of files) {
  const html = fs.readFileSync(path.join(dir, f), 'utf8');
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let m, idx = 0;
  while ((m = re.exec(html))) {
    idx++;
    const code = m[1];
    if (!code.trim()) continue;
    try { new Function(code); }
    catch (e) { bad++; console.log(`SYNTAX ERROR in ${f} block#${idx}: ${e.message}`); }
  }
}
console.log(bad === 0 ? `OK: all ${files.length} html inline scripts parse cleanly` : `FAILED: ${bad} errors`);
