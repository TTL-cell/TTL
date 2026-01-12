const METHODS_FALLBACK = ["GT", "baseline", "phoneme", "TTL"];
const AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"];

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return node;
}

// Some hosts block HEAD; fallback to GET.
async function exists(url) {
  try {
    const r = await fetch(url, { method: "HEAD" });
    if (r.ok) return true;
    if (r.status === 405) {
      const rg = await fetch(url, { method: "GET" });
      return rg.ok;
    }
    return false;
  } catch {
    return false;
  }
}

function buildAudioCell(url, ok) {
  const wrap = el("div", { class: "cell" });

  if (!ok) {
    wrap.appendChild(el("span", { class: "badge missing" }, ["Missing"]));
    return wrap;
  }

  const audio = el("audio", { controls: "controls", preload: "none" });
  audio.src = url;
  wrap.appendChild(audio);
  return wrap;
}

async function main() {
  const tbody = document.getElementById("tbody");

  let manifest;
  try {
    const res = await fetch("static/js/manifest.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`manifest.json fetch failed: ${res.status}`);
    manifest = await res.json();
  } catch (e) {
    tbody.innerHTML = "";
    const tr = document.createElement("tr");
    tr.appendChild(el("td", { colspan: "5" }, [`Failed to load manifest.json: ${e.message}`]));
    tbody.appendChild(tr);
    return;
  }

  const base = (manifest.baseAudioDir || "static/audio").replace(/\\/g, "/");
  const datasets = manifest.datasets || [];
  const methods = manifest.methods || METHODS_FALLBACK;

  // Render: dataset rows (one sample per dataset is recommended)
  for (const ds of datasets) {
    const dsId = ds.id;
    const dsLabel = ds.label || ds.id;

    // If multiple samples are listed, we render multiple rows (one per sample).
    const samples = (ds.samples && ds.samples.length) ? ds.samples : ["sample1.wav"];

    for (const sampleName of samples) {
      const tr = document.createElement("tr");

      // Dataset cell (NO filename shown)
      tr.appendChild(
        el("td", {}, [
          el("div", { class: "datasetTitle" }, [
            el("div", { class: "datasetName" }, [dsLabel])
          ])
        ])
      );

      // Method audio cells
      const urls = methods.map(m => `${base}/${dsId}/${m}/${sampleName}`);

      // Check existence (in parallel)
      const oks = await Promise.all(urls.map(u => exists(u)));

      for (let i = 0; i < methods.length; i++) {
        tr.appendChild(el("td", {}, [buildAudioCell(urls[i], oks[i])]));
      }

      tbody.appendChild(tr);
    }
  }
}

main();