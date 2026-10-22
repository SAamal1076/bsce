/* bsce workbench (front end). The server builds every command; this page only shows the forms, the commands the
   server returns, and the runs. No framework, no build step. */
"use strict";

const TOKEN = document.querySelector('meta[name="token"]').content;
const S = {
  meta: null, values: {}, jobs: [], choices: null, mode: "spyder",
  route: { view: "step", step: "env", task: null, id: null },
  plan: null, planSeq: 0, log: { id: null, next: 0, follow: true }, existsSeq: 0,
};

/* ---------- small helpers ---------- */
const $ = (sel, el = document) => el.querySelector(sel);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
async function api(path, body) {
  const opt = { headers: { "X-Token": TOKEN } };
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let j = {};
  try { j = await r.json(); } catch (e) { j = { error: "no reply from the app (is it still running?)" }; }
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const fileUrl = p => `/file?p=${encodeURIComponent(p)}&t=${encodeURIComponent(TOKEN)}`;
const store = {
  get(k, d) { try { const v = localStorage.getItem("bsce." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("bsce." + k, JSON.stringify(v)); } catch (e) { /* private window */ } },
};
function toast(msg, link) {
  const t = $("#toast");
  t.replaceChildren(msg, link ? h("a", { href: link.href, text: link.text }) : "");
  t.classList.add("show");
  clearTimeout(toast.tm);
  toast.tm = setTimeout(() => t.classList.remove("show"), 4200);
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = h("textarea", {}, text); document.body.append(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) { /* ignore */ } ta.remove();
  }
  toast("Copied");
}
const fmtDur = s => {
  if (s === null || s === undefined || !isFinite(s)) return "";
  s = Math.max(0, Math.round(s));
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${String(Math.floor(s % 3600 / 60)).padStart(2, "0")} min`;
  if (s >= 60) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
  return `${s} s`;
};
const fmtTime = t => t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
const STATUS = { queued: "Waiting in the queue", starting: "Starting", running: "Running", done: "Finished", failed: "Failed",
                 stopped: "Stopped", interrupted: "Interrupted (the app was closed)" };
const SCEN_LABEL = { sv_ula: "SV ULA", sv_upa: "SV UPA", deepmimo_ula: "DeepMIMO ULA", deepmimo_upa: "DeepMIMO UPA",
                     "3gpp_upa": "3GPP UPA", "3gpp_ula": "3GPP ULA" };

/* ---------- tasks and values ---------- */
function allTasks() {
  const out = {};
  for (const st of S.meta.steps) for (const t of st.tasks) out[t.id] = Object.assign({ step: st.id }, t);
  for (const t of S.meta.extra) out[t.id] = Object.assign({ step: "script" }, t);
  return out;
}
function defaults(task) {
  const v = {};
  for (const f of task.fields) v[f.key] = structuredClone(f.default);
  return v;
}
function valuesOf(task) {
  if (!S.values[task.id]) {
    const d = defaults(task), saved = store.get("values." + task.id, null);
    if (saved) {
      for (const f of task.fields) {
        if (!(f.key in saved)) continue;
        if (f.kind === "methods") {            // keep new methods of the code; take the saved ticks and depths
          const byKey = Object.fromEntries((saved[f.key] || []).map(r => [r.key, r]));
          d[f.key] = d[f.key].map(r => byKey[r.key] ? Object.assign({}, r, { on: byKey[r.key].on, T: byKey[r.key].T,
                                                                               set: byKey[r.key].set || "" }) : r);
        } else d[f.key] = saved[f.key];
      }
    }
    S.values[task.id] = d;
  }
  return S.values[task.id];
}
function setValue(task, key, val) {
  valuesOf(task)[key] = val;
  store.set("values." + task.id, S.values[task.id]);
  renderVisibility(task);
  schedulePlan(task);
}
function shown(f, v) {
  if (!f.show_if) return true;
  const [k, want] = f.show_if, cur = v[k];
  if (typeof want === "boolean") return !!cur === want;
  if (typeof want === "string" && want.startsWith("!")) return String(cur ?? "").trim() !== want.slice(1);
  return String(cur) === String(want);
}

/* ---------- routing ---------- */
function parseRoute() {
  const p = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (p[0] === "runs") return { view: "runs", id: p[1] || null };
  if (p[0] === "folders") return { view: "folders" };
  if (p[0] === "results") return { view: "results", id: p.slice(1).join("/") || null };
  if (p[0] === "script") return { view: "step", step: "script", task: "script" };
  const st = S.meta.steps.find(s => s.id === p[1]) || S.meta.steps[0];
  const task = st.tasks.find(t => t.id === p[2]) || st.tasks.find(t => t.id === store.get("tab." + st.id, "")) || st.tasks[0];
  return { view: "step", step: st.id, task: task.id };
}
function go(hash) { if (location.hash === hash) render(); else location.hash = hash; }
window.addEventListener("hashchange", () => render());

/* ---------- rail ---------- */
function stepState(stepId) {
  const tasks = stepId === "script" ? ["script"] : S.meta.steps.find(s => s.id === stepId).tasks.map(t => t.id);
  const js = S.jobs.filter(j => tasks.includes(j.task));
  if (js.some(j => j.status === "running" || j.status === "starting")) return "run";
  const last = js.find(j => ["done", "failed", "interrupted"].includes(j.status));
  if (!last) return "";
  return last.status === "done" ? "done" : "fail";
}
const ICON = {   // 18 x 18 line icons, drawn on the same square grid as the step cells
  runs: '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="2" width="5" height="5" fill="currentColor"/><rect x="11" y="2" width="5" height="5" fill="currentColor"/><rect x="2" y="11" width="5" height="5" fill="currentColor"/><rect x="11.8" y="11.8" width="3.4" height="3.4"/></svg>',
  nets: '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 5h14M2 9h14M2 13h14"/><circle cx="5" cy="5" r="1.4" fill="currentColor"/><circle cx="11" cy="9" r="1.4" fill="currentColor"/><circle cx="8" cy="13" r="1.4" fill="currentColor"/></svg>',
  results: '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2.5 2.5v13h13"/><path d="M5 5.5l3 4 3-2 4 5"/></svg>',
  script: '<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 5l4 4-4 4M9 14h6"/></svg>',
};
function bigIcon(name) {
  const el = h("div", { class: "bigno empty icon", "aria-hidden": "true" });
  el.innerHTML = ICON[name];
  return el;
}
function renderRail() {
  const r = S.route, rail = $("#rail");
  const item = (href, no, title, cur, state, icon) => {
    const cell = h("span", { class: `cellno ${state}${icon ? " icon" : ""}`, "aria-hidden": "true" }, icon ? "" : no);
    if (icon) cell.innerHTML = ICON[icon];
    return h("a", { href, "aria-current": cur ? "page" : null }, cell, h("span", {}, title));
  };
  rail.replaceChildren(
    h("div", { class: "rail-h" }, "Workflow"),
    ...S.meta.steps.map(st => item(`#/step/${st.id}`, st.n, st.title, r.view === "step" && r.step === st.id, stepState(st.id))),
    h("div", { class: "rail-h" }, "Watch and review"),
    item("#/runs", "", "Runs", r.view === "runs", S.jobs.some(j => j.status === "running") ? "run" : "", "runs"),
    item("#/folders", "", "Trained networks", r.view === "folders", "", "nets"),
    item("#/results", "", "Results", r.view === "results", "", "results"),
    h("div", { class: "rail-h" }, "More"),
    item("#/script", "", "Any script", r.view === "step" && r.step === "script", stepState("script") === "run" ? "run" : "", "script"),
  );
}

/* ---------- top bar ---------- */
function folderName(p) { const k = String(p || "").split(/[\\/]/).filter(Boolean); return k.slice(-2).join("\\"); }
function renderOthers() {
  const box = $("#others"), m = S.meta, list = (m.others || []).filter(o => o.kind === "bsce");
  box.hidden = !list.length;
  box.replaceChildren(...list.map(o => h("div", { class: "others-row", style: "display:contents" },
    h("p", {}, h("b", {}, "Another bsce workbench is still running "), `at ${o.url} `,
      o.root ? `(version ${o.version || "?"}, folder ${o.root}). ` : "(an older version, its folder unknown). ",
      `If the page that opened shows the old steps or the old runs, it was that one. This page is version ${m.version}, folder ${m.root}.`),
    h("a", { class: "btn small", href: o.url, target: "_blank", rel: "noopener" }, "Open it"),
    h("button", { class: "btn small", type: "button", onclick: () => stopOther(o) }, "Stop it"))));
}
async function stopOther(o) {
  if (!confirm(`Stop the bsce workbench at ${o.url}? Any step it is running stops too. Its trainings keep their checkpoints in its own outputs folder.`)) return;
  try {
    const r = await api("/api/stop_other", { port: o.port });
    toast(r.ok ? `Stopped the workbench at ${o.url}` : r.error);
  } catch (e) { toast(e.message); }
  try { S.meta = await api("/api/meta"); } catch (e) { /* kept */ }
  renderOthers();
}
function renderTop() {
  const m = S.meta;
  $("#pypath").textContent = `${m.python}  (${m.python_source})`;
  $("#ver").textContent = m.version ? `v${m.version}` : "";
  $("#where").textContent = folderName(m.root);
  document.title = `bsce workbench v${m.version || "?"}: ${folderName(m.root)}`;
  const run = S.jobs.filter(j => j.status === "running" || j.status === "starting"), q = S.jobs.filter(j => j.status === "queued");
  const parts = [];
  if (run.length) parts.push(`${run.length} running`);
  if (q.length) parts.push(`${q.length} waiting`);
  $("#queue-text").textContent = parts.length ? parts.join(", ") : (S.jobs.length ? "Nothing running" : "No runs yet");
  $("#queue").classList.toggle("busy", run.length > 0);
}

/* ---------- step pages ---------- */
function renderStep() {
  const r = S.route, T = allTasks(), task = T[r.task];
  const main = $("#main");
  const st = r.step === "script" ? { n: "", title: "Any script", intro: "Run any script of the bsce folder with your own arguments. Choose a script and keep --help to see its options first.", tasks: S.meta.extra }
                                  : S.meta.steps.find(s => s.id === r.step);
  store.set("tab." + r.step, task.id);
  const head = h("div", { class: "head" },
    st.n === "" ? bigIcon("script") : h("div", { class: "bigno", "aria-hidden": "true" }, st.n),
    h("div", {}, h("h1", {}, st.title), h("p", {}, st.intro)));
  const tabs = st.tasks.length > 1 ? h("div", { class: "tabs", role: "tablist" },
    st.tasks.map(t => h("button", { role: "tab", "aria-selected": String(t.id === task.id), type: "button",
                                    onclick: () => go(`#/step/${r.step}/${t.id}`) }, t.title))) : null;
  const form = h("div", { class: "form", id: "form" });
  const v = valuesOf(task);
  for (const f of task.fields) form.append(renderField(task, f, v));
  const reset = h("div", { class: "toolbar", style: "margin-top:18px" },
    task.fields.length ? h("button", { class: "btn small", type: "button", onclick: () => {
      S.values[task.id] = defaults(task); store.set("values." + task.id, S.values[task.id]); render(); } }, "Reset to defaults") : null);
  main.replaceChildren(head, tabs || "", task.note ? h("div", { class: "note" }, task.note) : "",
                       task.fields.length ? form : h("p", { class: "muted" }, "Nothing to set for this one."), reset);
  renderVisibility(task);
  renderCmdPanel(task);
  schedulePlan(task, 0);
  if (task.fields.some(f => ["network", "csv"].includes(f.kind))) loadChoices(task);
}

function fieldWrap(f, ...kids) {
  return h("div", { class: `fld k-${f.kind}${f.wide ? " wide" : ""}`, "data-key": f.key }, ...kids);
}
function renderField(task, f, v) {
  const id = `f-${task.id}-${f.key}`;
  const help = f.help ? h("div", { class: "help" }, f.help) : null;
  const lab = h("label", { class: "lab", for: id }, f.label);
  const set = val => setValue(task, f.key, val);
  switch (f.kind) {
    case "check":
      return fieldWrap(f, h("input", { type: "checkbox", id, checked: !!v[f.key], onchange: e => set(e.target.checked) }),
                       h("div", {}, h("label", { class: "lab", for: id }, f.label), help));
    case "select":
      return fieldWrap(f, lab, h("select", { id, onchange: e => set(e.target.value) },
        f.options.map(([val, txt]) => h("option", { value: val, selected: String(v[f.key]) === val }, txt))), help);
    case "multi":
      return fieldWrap(f, h("div", { class: "lab" }, f.label), h("div", { class: "chips", role: "group", "aria-label": f.label },
        f.options.map(([val, txt]) => h("label", { class: "chip" },
          h("input", { type: "checkbox", checked: (v[f.key] || []).includes(val), onchange: e => {
            const cur = new Set(v[f.key] || []); e.target.checked ? cur.add(val) : cur.delete(val);
            set(f.options.map(o => o[0]).filter(x => cur.has(x))); } }), txt))), help);
    case "scenarios": {
      const main = S.meta.main_scenarios, others = S.meta.scenarios.filter(s => !main.includes(s));
      const chip = s => h("label", { class: "chip" }, h("input", { type: "checkbox", checked: (v[f.key] || []).includes(s), onchange: e => {
        const cur = new Set(v[f.key] || []); e.target.checked ? cur.add(s) : cur.delete(s);
        set([...main, ...others].filter(x => cur.has(x))); } }), SCEN_LABEL[s] || s);
      const nOther = (v[f.key] || []).filter(s => others.includes(s)).length;
      return fieldWrap(f, h("div", { class: "lab" }, f.label), h("div", { class: "chips" }, main.map(chip)),
        others.length ? h("details", { class: "more", open: nOther > 0 }, h("summary", {}, `Other data sets (${others.length})`),
                          h("div", { class: "chips" }, others.map(chip))) : null, help);
    }
    case "methods": {
      const rows = v[f.key];
      const tb = h("tbody");
      rows.forEach((r, i) => {
        const tr = h("tr", { class: r.on ? "" : "off" },
          h("td", { style: "width:34px" }, h("input", { type: "checkbox", "aria-label": r.label, checked: r.on, onchange: e => {
            rows[i].on = e.target.checked; tr.className = e.target.checked ? "" : "off"; set(rows); } })),
          h("td", {}, r.label, h("span", { class: "key" }, r.key), r.desc ? h("div", { class: "mdesc" }, r.desc) : null),
          h("td", { class: "t" }, h("input", { type: "text", value: r.T, "aria-label": `${r.label} layers`, inputmode: "numeric",
            oninput: e => { rows[i].T = e.target.value; set(rows); } })),
          h("td", { class: "ms" }, h("input", { type: "text", value: r.set || "", "aria-label": `${r.label} settings`,
            placeholder: "key=value; key=value", spellcheck: "false", autocomplete: "off",
            oninput: e => { rows[i].set = e.target.value; set(rows); } })));
        tb.append(tr);
      });
      return fieldWrap(f, h("div", { class: "lab" }, f.label),
        h("table", { class: "meth" }, h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Method"), h("th", {}, "T (layers)"),
          h("th", {}, "Its own settings"))), tb), help);
    }
    case "lines":
      return fieldWrap(f, lab, h("textarea", { id, rows: Math.max(3, String(v[f.key] || "").split("\n").length + 1), placeholder: f.placeholder || "",
        spellcheck: "false", oninput: e => set(e.target.value) }, v[f.key] || ""), help);
    case "path": {
      const st = h("div", { class: "pathstate", "aria-live": "polite" });
      const inp = h("input", { type: "text", id, value: v[f.key] || "", placeholder: f.placeholder || "", spellcheck: "false", autocomplete: "off",
        oninput: e => { set(e.target.value); checkPath(e.target.value, st); } });
      if (v[f.key]) setTimeout(() => checkPath(v[f.key], st), 0);
      return fieldWrap(f, lab, inp, st, help);
    }
    case "network": case "csv": {
      const sel = h("select", { id, onchange: e => set(e.target.value), "data-choices": f.kind },
        h("option", { value: "" }, "loading…"));
      return fieldWrap(f, lab, sel, help);
    }
    case "script": {
      const out = h("pre", { class: "scripthelp", hidden: true });
      const show = h("button", { type: "button", class: "small", onclick: async () => {
        out.hidden = false; out.textContent = "loading…";
        try { out.textContent = (await api("/api/help", { script: v[f.key] })).text; } catch (e) { out.textContent = String(e); } } },
        "Show its options");
      return fieldWrap(f, lab, h("select", { id, onchange: e => { set(e.target.value); out.hidden = true; } },
        S.meta.scripts.map(s => h("option", { value: s, selected: v[f.key] === s }, s))), show, out, help);
    }
    default:
      return fieldWrap(f, lab, h("input", { type: "text", id, value: v[f.key] ?? "", placeholder: f.placeholder || "", spellcheck: "false",
        autocomplete: "off", inputmode: f.kind === "number" ? "decimal" : null, oninput: e => set(e.target.value) }), help);
  }
}
function renderVisibility(task) {
  const v = valuesOf(task);
  for (const f of task.fields) {
    const el = document.querySelector(`#form [data-key="${f.key}"]`);
    if (el) el.hidden = !shown(f, v);
  }
}
async function checkPath(val, el) {
  const seq = ++S.existsSeq;
  val = String(val || "").trim();
  if (!val) { el.textContent = ""; return; }
  clearTimeout(el._tm);
  el._tm = setTimeout(async () => {
    try {
      const r = await api("/api/exists", { paths: [val] });
      if (seq !== S.existsSeq) return;
      const n = r.found[0];
      el.className = "pathstate " + (n ? "ok" : "no");
      el.textContent = n ? (/[*?[]/.test(val) ? `${n} found` : "found") : "not found on this PC";
    } catch (e) { el.textContent = ""; }
  }, 300);
}
async function loadChoices(task) {
  try { S.choices = await api("/api/choices"); } catch (e) { return; }
  const v = valuesOf(task);
  for (const f of task.fields) {
    if (!["network", "csv"].includes(f.kind)) continue;
    const sel = document.querySelector(`#form [data-key="${f.key}"] select`);
    if (!sel) continue;
    const list = f.kind === "network" ? S.choices.networks : S.choices.csv;
    const none = f.kind === "network" ? "no trained network in outputs/ yet" : "no comparison yet (step 5 makes one)";
    sel.replaceChildren(h("option", { value: "" }, list.length ? "choose…" : none),
      ...list.map(p => h("option", { value: p, selected: v[f.key] === p }, p)));
  }
}

/* ---------- commands panel ---------- */
function schedulePlan(task, delay = 140) {
  clearTimeout(schedulePlan.tm);
  schedulePlan.tm = setTimeout(() => refreshPlan(task), delay);
}
async function refreshPlan(task) {
  const seq = ++S.planSeq;
  try {
    const p = await api("/api/plan", { task: task.id, values: valuesOf(task) });
    if (seq !== S.planSeq) return;
    S.plan = p;
  } catch (e) { S.plan = { ok: false, error: e.message, commands: [] }; }
  renderCommands(task);
}
function renderCmdPanel(task) {
  const cmd = $("#cmd");
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Show commands for" },
    ["spyder", "terminal"].map(m => h("button", { type: "button", "aria-pressed": String(S.mode === m), onclick: async () => {
      S.mode = m; renderCmdPanel(task); renderCommands(task);
      try { S.meta = await api("/api/settings", { show: m }); } catch (e) { /* kept for this page */ } } },
      m === "spyder" ? "Spyder" : "Terminal")));
  cmd.replaceChildren(
    h("div", { class: "cmd-h" }, h("h2", {}, "Commands"), seg),
    h("p", { class: "cmd-sub" }, "Exactly what runs when you press Add to queue. You can also copy them and run them yourself."),
    h("div", { id: "cmd-list" }),
    h("div", { class: "actions", id: "cmd-actions" }),
    h("div", { class: "latest", id: "latest", hidden: true }),
    h("div", { id: "preview" }));
  renderLatest();
}
function renderCommands(task) {
  const list = $("#cmd-list"), act = $("#cmd-actions");
  if (!list) return;
  const p = S.plan || { ok: false, commands: [] };
  if (!p.ok) {
    list.replaceChildren(h("div", { class: "cmd-err" }, h("strong", {}, "Fix this first: "), p.error || "…"));
  } else {
    const many = p.commands.length > 1;
    list.replaceChildren(...p.commands.map((c, i) => {
      const text = S.mode === "spyder" ? c.spyder : c.terminal;
      return h("div", { class: "cmdbox" },
        h("div", { class: "clabel" }, h("span", {}, many ? h("span", { class: "n" }, `${i + 1}`) : "", c.label),
          h("button", { class: "btn small ghost", type: "button", onclick: () => copy(text), "aria-label": `Copy command ${i + 1}` }, "Copy")),
        h("pre", {}, text),
        h("div", { class: "where" }, "Run it in: " + (S.mode === "spyder" ? c.where_spyder : c.where_terminal)));
    }));
  }
  const all = () => p.commands.map(c => S.mode === "spyder" ? c.spyder : c.terminal).join("\n");
  act.replaceChildren(
    h("button", { class: "btn primary", type: "button", disabled: !p.ok, onclick: () => submit(task, false),
                  title: "Runs after the runs before it (one at a time, for the GPU)" }, "Add to queue"),
    h("button", { class: "btn", type: "button", disabled: !p.ok, onclick: () => submit(task, true),
                  title: "Starts now, next to the queue (e.g. a CPU step while the GPU trains)" }, "Run now, in parallel"),
    task.dryrun ? h("button", { class: "btn", type: "button", disabled: !p.ok, onclick: () => dryrun(task) }, "Preview the plan") : "",
    p.ok && p.commands.length > 1 ? h("button", { class: "btn ghost", type: "button", onclick: () => copy(all()) }, "Copy all") : "");
}
async function submit(task, parallel) {
  try {
    const j = await api("/api/run", { task: task.id, values: valuesOf(task), parallel });
    S.jobs.unshift(j);
    renderTop(); renderRail(); renderLatest();
    toast(parallel ? "Started" : "Added to the queue", { href: `#/runs/${j.id}`, text: "Watch it" });
  } catch (e) { toast("Not started: " + e.message); }
}
async function dryrun(task) {
  const box = $("#preview");
  box.replaceChildren(h("p", { class: "muted" }, "Planning (dry run, nothing is trained)…"));
  try {
    const r = await api("/api/dryrun", { task: task.id, values: valuesOf(task) });
    box.replaceChildren(h("div", { class: "cmd-h", style: "margin-top:10px" }, h("h2", {}, "Plan"),
      h("button", { class: "btn small ghost", type: "button", onclick: () => box.replaceChildren() }, "Close")),
      h("p", { class: "cmd-sub" }, "Lines with >>> will run; lines with --- are already done and are skipped."),
      h("pre", { class: "preview" }, r.text));
  } catch (e) { box.replaceChildren(h("div", { class: "cmd-err" }, e.message)); }
}
function renderLatest() {
  const el = $("#latest");
  if (!el || S.route.view !== "step") return;
  const j = S.jobs.find(x => x.task === S.route.task);
  if (!j) { el.hidden = true; return; }
  el.hidden = false;
  el.replaceChildren(h("h3", {}, "Last run of this task"), jobBrief(j, false),
    h("a", { href: `#/runs/${j.id}` }, "Open the run and its console output"));
}

/* ---------- progress (the beam grid) ---------- */
function beamGrid(prog, big) {
  const g = h("div", { class: "grid" + (big ? " big" : ""), role: "img",
    "aria-label": `${prog.cells.filter(c => c.state === "done" || c.state === "skip").length} of ${prog.cells.length} steps finished` });
  for (const c of prog.cells) g.append(h("i", { class: c.state, title: `${c.label} (${({ todo: "to do", run: "running", done: "finished", fail: "failed", skip: "done earlier, skipped" })[c.state]})` }));
  return g;
}
function jobBrief(j, big) {
  const p = j.progress || { cells: [] };
  const n = p.cells.length, d = p.cells.filter(c => c.state === "done" || c.state === "skip").length;
  const running = j.status === "running";
  const cur = p.cur !== null && p.cur !== undefined ? p.cells[p.cur] : null;
  return h("div", {},
    h("div", { class: "now" }, h("span", { class: `status ${j.status}` }, STATUS[j.status] || j.status),
      n ? `, ${d} of ${n} steps finished` : "", j.status === "queued" ? "" : (j.started ? `, ${fmtDur((j.ended || Date.now() / 1000) - j.started)}` : "")),
    n ? beamGrid(p, big) : null,
    running && cur ? h("div", { class: "now" }, cur.label) : null,
    running && p.detail ? h("div", { class: "now" }, p.detail, p.eta ? `, about ${fmtDur(p.eta)} left in this step` : "") : null,
    running && p.frac !== null && p.frac !== undefined ? h("div", { class: "bar" }, h("b", { style: `width:${(100 * p.frac).toFixed(1)}%` })) : null,
    running && p.metric ? h("div", { class: "now" }, h("span", { class: "metric" }, p.metric)) : null,
    j.error ? h("div", { class: "cmd-err", style: "margin-top:8px" }, j.error) : null);
}

/* ---------- runs ---------- */
function defaultJob(id) {
  return S.jobs.find(j => j.id === id) || S.jobs.find(j => j.status === "running") || S.jobs[0];
}
function renderRuns() {
  const main = $("#main"), r = S.route;
  const sel = defaultJob(r.id);
  const head = h("div", { class: "head" }, bigIcon("runs"),
    h("div", {}, h("h1", {}, "Runs"), h("p", {}, "Everything started from this app. The queue runs one at a time; closing the browser does not stop a run, closing the app window does.")));
  if (!S.jobs.length) {
    main.replaceChildren(head, h("div", { class: "blank" }, "No runs yet. Choose a step on the left, check its commands and press Add to queue."));
    return;
  }
  const list = h("div", { class: "joblist", role: "list" }, S.jobs.map(j => h("a", { href: `#/runs/${j.id}`, role: "listitem", "aria-current": String(sel && j.id === sel.id) },
    h("span", { class: `dot ${j.status}`, title: STATUS[j.status] }),
    h("span", { class: "t" }, j.title, h("div", { class: "d" }, `${STATUS[j.status] || j.status}, ${fmtTime(j.started || j.created)}`)),
    h("span", { class: "d" }, j.started ? fmtDur((j.ended || Date.now() / 1000) - j.started) : ""))));
  const detail = h("div", { id: "jobdetail" });
  main.replaceChildren(head, h("div", { class: "runs" }, list, detail));
  if (sel) renderJobDetail(sel, true);
}
function renderJobDetail(j, fresh) {
  const box = $("#jobdetail");
  if (!box) return;
  const active = ["running", "starting", "queued"].includes(j.status);
  const top = h("div", { class: "panel", id: "jobtop" },
    h("div", { class: "jobhead" }, h("h2", {}, j.title),
      h("div", { class: "toolbar", style: "margin:0" },
        active ? h("button", { class: "btn danger small", type: "button", onclick: () => stopJob(j) }, j.status === "queued" ? "Remove from the queue" : "Stop") : null,
        !active ? h("button", { class: "btn small", type: "button", onclick: () => rerun(j), title: "Finished steps are skipped and an interrupted training resumes" }, "Run again") : null,
        !active ? h("button", { class: "btn small ghost", type: "button", onclick: () => removeJob(j) }, "Remove from the list") : null)),
    h("div", { class: "kv" }, h("span", {}, `added ${fmtTime(j.created)}`), j.started ? h("span", {}, `started ${fmtTime(j.started)}`) : null,
      j.ended ? h("span", {}, `ended ${fmtTime(j.ended)}`) : null, j.rc !== null && j.rc !== undefined ? h("span", {}, `exit code ${j.rc}`) : null,
      j.parallel ? h("span", {}, "ran in parallel") : null),
    h("div", { style: "margin-top:10px" }, jobBrief(j, true)),
    (j.progress && j.progress.cells.length) ? h("div", { class: "legend", style: "margin-top:6px" },
      h("span", { class: "l-done" }, "finished"), h("span", { class: "l-skip" }, "done earlier, skipped"), h("span", { class: "l-run" }, "running"),
      h("span", { class: "l-fail" }, "failed"), h("span", {}, "to do")) : null,
    h("details", {}, h("summary", { class: "muted", style: "margin-top:10px;cursor:pointer" }, `Commands (${j.n_cmds})`),
      h("ol", { class: "cmdlist" }, j.commands.map((c, i) => h("li", { class: i === j.cmd_i && active ? "cur" : "" }, j.labels[i] ? `${j.labels[i]}: ` : "", c)))));
  if (fresh || !$("#log")) {
    S.log = { id: j.id, next: 0, follow: true };
    const follow = h("input", { type: "checkbox", checked: true, onchange: e => { S.log.follow = e.target.checked; } });
    const logp = h("div", { class: "panel" }, h("div", { class: "loghead", style: "margin-top:0" }, h("h3", { style: "font-size:14px" }, "Console output"),
      h("div", { class: "toolbar", style: "margin:0" }, h("label", {}, follow, "Follow the end"),
        h("a", { href: fileUrl(`app/runs/${j.id}.log`), target: "_blank", rel: "noopener" }, "Full log file"))),
      h("div", { class: "log", id: "log", tabindex: "0", "aria-label": "Console output" }));
    box.replaceChildren(top, logp);
    pollLog();
  } else {
    box.replaceChild(top, $("#jobtop"));
  }
}
function classify(ln) {
  if (ln.startsWith("$ ") || ln.startsWith(">>> ")) return "c";
  if (/^!!!|Traceback|Error|error:|FAILED|step failed/.test(ln)) return "e";
  if (ln.startsWith("###") || ln.startsWith("---") || ln.startsWith("########")) return "s";
  return "";
}
async function pollLog() {
  const L = S.log, el = $("#log");
  if (!L.id || !el) return;
  let r;
  try { r = await api(`/api/job?id=${encodeURIComponent(L.id)}&from=${L.next}`); } catch (e) { return; }
  if (S.log !== L) return;
  if (r.lines.length) {
    const frag = document.createDocumentFragment();
    for (const ln of r.lines) { const c = classify(ln); frag.append(c ? h("span", { class: c }, ln + "\n") : document.createTextNode(ln + "\n")); }
    el.append(frag);
    while (el.childNodes.length > 6000) el.removeChild(el.firstChild);
    if (L.follow) el.scrollTop = el.scrollHeight;
  }
  L.next = r.next;
}
async function stopJob(j) {
  if (j.status !== "queued" && !confirm("Stop this run? Its training checkpoint is kept: Run again resumes it.")) return;
  try { await api("/api/stop", { id: j.id }); toast("Stopping"); } catch (e) { toast(e.message); }
  pollJobs();
}
async function rerun(j) {
  try { const n = await api("/api/rerun", { id: j.id }); S.jobs.unshift(n); go(`#/runs/${n.id}`); toast("Added to the queue"); }
  catch (e) { toast(e.message); }
}
async function removeJob(j) {
  try { await api("/api/remove", { id: j.id }); S.jobs = S.jobs.filter(x => x.id !== j.id); go("#/runs"); } catch (e) { toast(e.message); }
}

/* ---------- trained networks ---------- */
async function renderFolders() {
  const main = $("#main");
  const head = h("div", { class: "head" }, bigIcon("nets"),
    h("div", {}, h("h1", {}, "Trained networks"), h("p", {}, "Every run folder of outputs/ (or another output folder): its bands, its per-layer snapshots and the best validation NMSE of its log.")));
  let dirs = ["outputs"];
  try { dirs = (await api("/api/choices")).output_dirs; } catch (e) { /* keep outputs */ }
  const curDir = dirs.includes(store.get("outdir", "outputs")) ? store.get("outdir", "outputs") : (dirs[0] || "outputs");
  const bar = h("div", { class: "toolbar" },
    dirs.length > 1 ? h("label", { class: "lab", style: "display:flex;align-items:center;gap:8px" }, "Folder",
      h("select", { style: "width:auto", onchange: e => { store.set("outdir", e.target.value); renderFolders(); } },
        dirs.map(x => h("option", { value: x, selected: x === curDir }, x)))) : null,
    h("button", { class: "btn small", type: "button", onclick: () => renderFolders() }, "Refresh"));
  main.replaceChildren(head, bar, h("p", { class: "muted" }, "Reading the folders…"));
  let d;
  try { d = await api(`/api/outputs?dir=${encodeURIComponent(curDir)}`); } catch (e) { main.replaceChildren(head, h("div", { class: "cmd-err" }, e.message)); return; }
  if (S.route.view !== "folders") return;
  if (!d.length) { main.replaceChildren(head, bar, h("div", { class: "blank" }, "No trained network yet. Step 4 trains them; step 3 makes the deep runs.")); return; }
  const blocks = d.map(sc => [h("h2", { class: "sc-h" }, sc.scenario),
    h("table", { class: "data" }, h("thead", {}, h("tr", {}, ["Network", "State", "Bands", "Layers saved", "Fine-tune iterations", "Best validation (dB)", "Changed"].map((x, i) => h("th", { class: i >= 3 && i <= 4 ? "num" : "" }, x)))),
      h("tbody", {}, sc.runs.map(r => h("tr", {},
        h("td", {}, h("strong", {}, r.name), r.readme ? h("div", { class: "help" }, r.readme) : null),
        h("td", {}, h("span", { class: "tag " + (r.state === "trained" ? "trained" : String(r.state).startsWith("in progress") ? "prog" : "") }, r.state)),
        h("td", {}, r.bands.map(b => h("span", { class: "tag" }, b))),
        h("td", { class: "num" }, r.layers || ""), h("td", { class: "num" }, r.ft_iters ? r.ft_iters.toLocaleString() : ""),
        h("td", {}, Object.entries(r.best || {}).map(([k, v]) => h("div", { class: "mono", style: "white-space:nowrap" }, `${k}  ${v}`))),
        h("td", { class: "d muted" }, fmtTime(r.mtime))))))]);
  main.replaceChildren(head, bar, ...blocks.flat());
}

/* ---------- results ---------- */
function mdToHtml(md) {
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = s => esc(s).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [], L = md.split(/\r?\n/);
  for (let i = 0; i < L.length; i++) {
    const ln = L[i];
    if (/^\s*\|/.test(ln)) {
      const rows = [];
      while (i < L.length && /^\s*\|/.test(L[i])) rows.push(L[i++]);
      i--;
      const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const body = rows.filter(r => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      out.push("<table><thead><tr>" + cells(body[0]).map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        body.slice(1).map(r => "<tr>" + cells(r).map(c => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
    } else if (/^#{1,6} /.test(ln)) out.push(`<h3>${inline(ln.replace(/^#+ /, ""))}</h3>`);
    else if (ln.trim()) out.push(`<p>${inline(ln)}</p>`);
  }
  return out.join("\n");
}
async function renderResults() {
  const main = $("#main");
  const head = h("div", { class: "head" }, bigIcon("results"),
    h("div", {}, h("h1", {}, "Results"), h("p", {}, "The figures and tables of results/<name>/, made by Compare (step 5) and the depth curves (step 3).")));
  let list;
  try { list = await api("/api/results"); } catch (e) { main.replaceChildren(head, h("div", { class: "cmd-err" }, e.message)); return; }
  if (S.route.view !== "results") return;
  if (!list.length) { main.replaceChildren(head, h("div", { class: "blank" }, "No results yet. Step 5 (Compare and plot) writes them.")); return; }
  const cur = list.find(r => r.name === S.route.id) || list[0];
  const picker = h("div", { class: "tabs" }, list.map(r => h("button", { type: "button", "aria-selected": String(r.name === cur.name),
    onclick: () => go(`#/results/${encodeURIComponent(r.name)}`) }, r.name)));
  const pngs = cur.files.filter(f => f.endsWith(".png"));
  const top = pngs.filter(f => f.split("/").length === 3), sub = pngs.filter(f => f.split("/").length > 3);
  const fig = f => h("figure", {}, h("a", { href: fileUrl(f), target: "_blank", rel: "noopener" }, h("img", { src: fileUrl(f), alt: f, loading: "lazy" })),
                     h("figcaption", {}, f.split("/").slice(2).join("/")));
  const md = h("div", { class: "md" });
  const summary = cur.files.find(f => f.split("/").length === 3 && f.endsWith("summary.md")) || cur.files.find(f => f.endsWith(".md"));
  main.replaceChildren(head, picker, h("div", { class: "kv", style: "margin:-8px 0 14px" }, `results/${cur.name}, written ${fmtTime(cur.mtime)}`),
    summary ? md : "", top.length ? h("div", { class: "figs" }, top.map(fig)) : "",
    sub.length ? h("details", { style: "margin-top:16px" }, h("summary", { style: "cursor:pointer" }, `Per-scenario figures (${sub.length})`),
                   h("div", { class: "figs", style: "margin-top:10px" }, sub.map(fig))) : "",
    h("h2", { class: "sc-h" }, "Files"),
    h("ul", { class: "files" }, cur.files.map(f => h("li", {}, h("a", { href: fileUrl(f), target: "_blank", rel: "noopener" }, f.split("/").slice(2).join("/"))))));
  if (summary) {
    try { const r = await fetch(fileUrl(summary)); md.innerHTML = mdToHtml(await r.text()); } catch (e) { md.textContent = ""; }
  }
}

/* ---------- settings ---------- */
function openSettings() {
  const d = $("#settings"), f = $("#settings-form"), s = S.meta.settings;
  for (const k of ["env_name", "python", "conda", "device", "show"]) f.elements[k].value = s[k] || "";
  if (!s.env_name) f.elements.env_name.value = "bsce";
  $("#detected").replaceChildren(
    h("dt", {}, "Steps run with"), h("dd", {}, `${S.meta.python} (${S.meta.python_source})`),
    h("dt", {}, "conda"), h("dd", {}, S.meta.conda || "not found: start the app from the Anaconda Prompt"),
    h("dt", {}, "This app"), h("dd", {}, S.meta.app_python),
    h("dt", {}, "bsce folder"), h("dd", {}, S.meta.root),
    h("dt", {}, "Version"), h("dd", {}, S.meta.version));
  d.showModal();
}
$("#open-settings").addEventListener("click", openSettings);
$("#pyline").addEventListener("click", openSettings);
$("#settings-form").addEventListener("submit", async e => {
  if (e.submitter && e.submitter.value === "save") {
    e.preventDefault();
    const f = e.target, body = {};
    for (const k of ["env_name", "python", "conda", "device", "show"]) body[k] = f.elements[k].value;
    try { S.meta = await api("/api/settings", body); S.mode = S.meta.settings.show || "spyder"; S.values = {}; $("#settings").close(); render(); toast("Saved"); }
    catch (err) { toast(err.message); }
  }
});

/* ---------- polling ---------- */
async function pollJobs() {
  try { S.jobs = await api("/api/jobs"); } catch (e) { $("#queue-text").textContent = "The app is not answering"; return; }
  renderTop(); renderRail(); renderLatest();
  if (S.route.view === "runs") {
    const sel = defaultJob(S.route.id);
    if (sel && $("#jobdetail")) {
      if (S.log.id !== sel.id) renderRuns(); else renderJobDetail(sel, false);
      const items = document.querySelectorAll(".joblist a");
      if (items.length !== S.jobs.length) renderRuns();
      else S.jobs.forEach((j, i) => { const dot = items[i].querySelector(".dot"); if (dot) dot.className = `dot ${j.status}`;
        const dd = items[i].querySelectorAll(".d"); if (dd[0]) dd[0].textContent = `${STATUS[j.status] || j.status}, ${fmtTime(j.started || j.created)}`;
        if (dd[1]) dd[1].textContent = j.started ? fmtDur((j.ended || Date.now() / 1000) - j.started) : ""; });
    } else if (S.jobs.length && !$("#jobdetail")) renderRuns();
    pollLog();
  }
}

/* ---------- GPU (nvidia-smi) ---------- */
async function pollGpu() {
  let g;
  try { g = await api("/api/gpu"); } catch (e) { return; }
  const el = $("#gpu");
  if (!g.gpus || !g.gpus.length) { el.hidden = true; return; }
  const x = g.gpus[0];
  el.hidden = false;
  el.replaceChildren(h("span", {}, "GPU"), h("span", { class: "meter", "aria-hidden": "true" }, h("b", { style: `width:${x.util}%` })),
    `${x.util}%, ${(x.mem_used / 1024).toFixed(1)} of ${(x.mem_total / 1024).toFixed(1)} GB, ${x.temp} °C`);
  el.title = `${x.name}: load, memory and temperature (nvidia-smi)`;
}

/* ---------- main render ---------- */
function render() {
  S.route = parseRoute();
  const shell = document.querySelector(".shell");
  shell.classList.toggle("wide", S.route.view !== "step");
  renderRail(); renderTop();
  if (S.route.view === "step") renderStep();
  else if (S.route.view === "runs") renderRuns();
  else if (S.route.view === "folders") renderFolders();
  else if (S.route.view === "results") renderResults();
  $("#main").focus({ preventScroll: true });
}
(async function init() {
  try { S.meta = await api("/api/meta"); } catch (e) {
    $("#main").replaceChildren(h("div", { class: "cmd-err" }, "The app did not answer: " + e.message)); return;
  }
  S.mode = S.meta.settings.show || "spyder";
  try { S.jobs = await api("/api/jobs"); } catch (e) { S.jobs = []; }
  render();
  renderOthers();
  setInterval(pollJobs, 1500);
  pollGpu();
  setInterval(pollGpu, 4000);
})();
