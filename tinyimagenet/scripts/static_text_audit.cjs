// Static text/file checks only. This script never imports or executes Python.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const all = [];
function walk(dir) {
  for (const ent of fs.readdirSync(dir, {withFileTypes: true})) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p); else all.push(p);
  }
}
walk(root);
function text(p) { return fs.readFileSync(p, 'utf8').replace(/^\uFEFF/, ''); }
function assert(value, message) { if (!value) throw new Error(message); }
function delimiters(source, file) {
  let stack = [], quote = null, triple = false, line = 1;
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (c === '\n') line++;
    if (quote) {
      if (c === '\\') { i++; continue; }
      if (triple && source.slice(i, i + 3) === quote.repeat(3)) { i += 2; quote = null; }
      else if (!triple && c === quote) quote = null;
      continue;
    }
    if (c === '#') { while (i < source.length && source[i] !== '\n') i++; line++; continue; }
    if (c === '"' || c === "'") {
      quote = c; triple = source.slice(i, i + 3) === c.repeat(3);
      if (triple) i += 2;
      continue;
    }
    if ('([{'.includes(c)) stack.push([c, line]);
    if (')]}'.includes(c)) {
      const opened = stack.pop();
      assert(opened && '([{'.indexOf(opened[0]) === ')]}'.indexOf(c), `${file}:${line} delimiter mismatch`);
    }
  }
  assert(!quote && !stack.length, `${file}: unclosed string/delimiter`);
}
const python = all.filter(p => p.endsWith('.py'));
for (const p of python) delimiters(text(p), p);
const json = all.filter(p => p.endsWith('.json') && !p.includes(path.sep + 'legacy_inputs' + path.sep));
for (const p of json) JSON.parse(text(p));
const notebook = JSON.parse(text(path.join(root, 'notebooks/Kaggle_Dual_T4_TinyImageNet_NC_CL.ipynb')));
for (const c of notebook.cells.filter(c => c.cell_type === 'code')) {
  assert(c.execution_count === null && c.outputs.length === 0, 'Notebook has executed results');
  delimiters(c.source.join(''), 'notebook cell');
}
for (const entry of JSON.parse(text(path.join(root, 'evidence/reused_source_hashes.json')))) {
  const hash = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, entry.path))).digest('hex');
  assert(hash === entry.sha256.toLowerCase(), 'Changed reused source: ' + entry.path);
}
const plan = JSON.parse(text(path.join(root, 'run_plan.json')));
assert(plan.length === 12 && new Set(plan.map(x => x.run_id)).size === 12, 'Wrong run plan');
assert(plan.every(x => ['finetune', 'lwf', 'etfs', 'lwf_etf'].includes(x.method) && [42,43,44].includes(x.seed)), 'Unexpected principal job');
const banned = all.filter(p => /\.(pdf|docx|pt|pth|pyc|zip)$/i.test(p));
assert(!banned.length, 'Non-source artifacts in delivery: ' + banned.join(', '));
for (const dir of ['outputs', 'paper_outputs']) {
  const files = all.filter(p => p.startsWith(path.join(root, dir) + path.sep));
  assert(files.length === 1 && path.basename(files[0]) === 'README.md', 'New outputs must be empty');
}
console.log(JSON.stringify({files: all.length, python_files_lexically_checked: python.length, json_files_checked: json.length, notebook_code_cells: notebook.cells.filter(c => c.cell_type === 'code').length, principal_jobs: plan.length, executed_python: false, limitation: 'Lexical/structural checks, not a Python parser or runtime test.'}, null, 2));
