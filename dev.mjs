// dev.mjs — one-command launcher for the whole stack.
//
// Starts BOTH processes together so you never have to juggle terminals:
//   1. FDTD engine server  (js_fdtd, http://localhost:4000)  — runs the sims
//   2. Next.js frontend    (visual,  http://localhost:3000)  — the dashboard
//
// The compute backend (auto / wasm-cpu / webgpu / cuda) is chosen entirely in
// the UI's "Compute Engine" selector — it's POSTed to the engine server per run,
// so nothing here needs to change when you switch engines.
//
// Zero external deps — uses only Node's child_process (matches the project's
// no-dependency ethos). Ctrl+C tears both children down.

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));
const isWin = process.platform === 'win32';
const npm = isWin ? 'npm.cmd' : 'npm';

// ANSI colors so the interleaved output stays readable.
const COLORS = { engine: '\x1b[36m', web: '\x1b[35m', reset: '\x1b[0m' };

const procs = [
  { tag: 'engine', color: COLORS.engine, cwd: 'js_fdtd', args: ['run', 'serve'] },
  { tag: 'web',    color: COLORS.web,    cwd: 'visual',  args: ['run', 'dev'] },
];

const children = [];
let shuttingDown = false;

function prefix(tag, color, chunk) {
  const label = `${color}[${tag}]${COLORS.reset}`;
  for (const line of chunk.toString().split(/\r?\n/)) {
    if (line.length) process.stdout.write(`${label} ${line}\n`);
  }
}

for (const p of procs) {
  const child = spawn(npm, p.args, {
    cwd: path.join(root, p.cwd),
    shell: isWin,            // npm.cmd needs a shell on Windows
    env: process.env,
  });
  child.stdout.on('data', (d) => prefix(p.tag, p.color, d));
  child.stderr.on('data', (d) => prefix(p.tag, p.color, d));
  child.on('exit', (code) => {
    prefix(p.tag, p.color, `exited with code ${code}`);
    // If one dies, bring the other down too so you're not left half-running.
    if (!shuttingDown) shutdown(code ?? 0);
  });
  children.push(child);
}

function shutdown(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const c of children) {
    if (c.exitCode === null) c.kill('SIGTERM');
  }
  setTimeout(() => process.exit(code), 500);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

console.log('Starting FDTD engine (:4000) + dashboard (:3000) — pick the compute engine in the UI.');
console.log('Press Ctrl+C to stop both.\n');
