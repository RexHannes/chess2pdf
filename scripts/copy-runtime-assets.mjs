import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const nodeModules = join(root, "node_modules");
const copied = [];
const warnings = [];
const requiredAssets = [];

function warn(message) {
  warnings.push(message);
}

function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

function fileSize(path) {
  try {
    return statSync(path).size;
  } catch {
    return 0;
  }
}

function copyAsset(source, target, { required = true, minBytes = 1_024 } = {}) {
  if (!source || !existsSync(source)) {
    const message = `[runtime-assets] Missing source: ${source || "not found"}`;
    if (required) {
      warn(message);
    } else {
      console.warn(message);
    }
    return false;
  }

  const size = fileSize(source);
  if (size < minBytes) {
    warn(`[runtime-assets] Source is unexpectedly small (${size} bytes): ${source}`);
    return false;
  }

  ensureDir(dirname(target));
  copyFileSync(source, target);
  copied.push(`${relative(source)} -> ${relative(target)}`);
  if (required) {
    requiredAssets.push({ path: target, minBytes });
  }
  return true;
}

function copyAssetsFromDir(sourceDir, targetDir, predicate, { required = true, minBytes = 1_024 } = {}) {
  if (!existsSync(sourceDir)) {
    if (required) warn(`[runtime-assets] Missing directory: ${sourceDir}`);
    return 0;
  }

  ensureDir(targetDir);
  let count = 0;
  for (const fileName of readdirSync(sourceDir)) {
    const source = join(sourceDir, fileName);
    if (!statSync(source).isFile() || !predicate(fileName)) {
      continue;
    }
    if (copyAsset(source, join(targetDir, fileName), { required, minBytes })) {
      count += 1;
    }
  }
  return count;
}

function findFiles(startDir, predicate, maxDepth = 6) {
  const results = [];
  function walk(dir, depth) {
    if (depth < 0 || !existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath, depth - 1);
      } else if (entry.isFile() && predicate(entry.name, fullPath)) {
        results.push(fullPath);
      }
    }
  }
  walk(startDir, maxDepth);
  return results.sort((a, b) => a.localeCompare(b));
}

function pickPreferred(paths, preferredPattern) {
  return paths.find((path) => preferredPattern.test(path)) ?? paths[0];
}

function relative(path) {
  return path.startsWith(root) ? path.slice(root.length + 1) : path;
}

function copyPdfJsWorker() {
  const candidates = [
    join(nodeModules, "pdfjs-dist", "build", "pdf.worker.min.mjs"),
    join(nodeModules, "pdfjs-dist", "legacy", "build", "pdf.worker.min.mjs"),
  ];
  copyAsset(candidates.find(existsSync), join(root, "public", "pdfjs", "pdf.worker.min.mjs"));
}

function copyTesseractWorker() {
  const candidates = [
    join(nodeModules, "tesseract.js", "dist", "worker.min.js"),
    ...findFiles(join(nodeModules, "tesseract.js"), (name) => name === "worker.min.js", 4),
  ];
  copyAsset(candidates.find(existsSync), join(root, "public", "tesseract", "worker.min.js"));
}

function copyTesseractCore() {
  const coreRoot = join(nodeModules, "tesseract.js-core");
  const files = findFiles(coreRoot, (name) => /^tesseract-core.*\.(?:js|wasm)$/.test(name), 4);
  if (files.length === 0) {
    warn(`[runtime-assets] Could not find Tesseract core files under ${coreRoot}`);
    return;
  }
  for (const source of files) {
    copyAsset(source, join(root, "public", "tesseract", basename(source)));
  }
}

function copyTesseractLanguageData() {
  const langRoot = join(nodeModules, "@tesseract.js-data", "eng");
  const files = findFiles(langRoot, (name) => name === "eng.traineddata.gz", 5);
  copyAsset(pickPreferred(files, /eng\.traineddata\.gz$/), join(root, "public", "tessdata", "eng.traineddata.gz"), { minBytes: 100_000 });
}

function copyOnnxRuntime() {
  const count = copyAssetsFromDir(
    join(nodeModules, "onnxruntime-web", "dist"),
    join(root, "public", "ort"),
    (name) => /^ort-.*\.(?:mjs|js|wasm)$/.test(name),
  );
  if (count === 0) {
    warn("[runtime-assets] No ONNX Runtime Web WASM files were copied.");
  }
}

function copyStockfish() {
  const stockfishRoot = join(nodeModules, "stockfish");
  const jsFiles = findFiles(stockfishRoot, (name) => /^stockfish-.*(?:lite|single).*\.js$/.test(name), 5);
  const wasmFiles = findFiles(stockfishRoot, (name) => /^stockfish-.*(?:lite|single).*\.wasm$/.test(name), 5);
  const js = pickPreferred(jsFiles, /stockfish-18-lite-single\.js$/);
  const wasm = pickPreferred(wasmFiles, /stockfish-18-lite-single\.wasm$/);

  copyAsset(js, join(root, "public", "stockfish", "stockfish-18-lite-single.js"));
  copyAsset(wasm, join(root, "public", "stockfish", "stockfish-18-lite-single.wasm"), { minBytes: 100_000 });
}

function verifyRequiredAssets() {
  for (const asset of requiredAssets) {
    const size = fileSize(asset.path);
    if (size < asset.minBytes) {
      warn(`[runtime-assets] Required output is missing or empty: ${relative(asset.path)} (${size} bytes)`);
    }
  }
}

copyPdfJsWorker();
copyTesseractWorker();
copyTesseractCore();
copyTesseractLanguageData();
copyOnnxRuntime();
copyStockfish();
verifyRequiredAssets();

if (copied.length > 0) {
  console.log(`[runtime-assets] Copied ${copied.length} browser runtime asset${copied.length === 1 ? "" : "s"}.`);
}
for (const entry of copied) {
  console.log(`  ${entry}`);
}

if (warnings.length > 0) {
  for (const warning of warnings) {
    console.error(warning);
  }
  process.exitCode = 1;
}
