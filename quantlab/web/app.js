/* quantlab UI — vanilla JS, no dependencies, no external requests. */
'use strict';

const api = {
  async call(path, opts = {}) {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch { /* empty body is fine */ }
    if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${res.statusText}`);
    return data;
  },
  get: (p) => api.call(p),
  post: (p, body) => api.call(p, { method: 'POST', body }),
  del: (p) => api.call(p, { method: 'DELETE' }),
};

const state = {
  portfolio: null, accounts: [], strategies: [], quote: null,
  side: 'buy', view: 'dashboard', backtest: null, history: null,
};

/* ------------------------------------------------------------- formatting */
const money = (v, d = 2) =>
  (v < 0 ? '-' : '') + '$' + Math.abs(Number(v) || 0).toLocaleString('en-US',
    { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 2) => `${(Number(v) || 0) >= 0 ? '' : '-'}${Math.abs((Number(v) || 0) * 100).toFixed(d)}%`;
const num = (v, d = 2) => (Number(v) || 0).toLocaleString('en-US',
  { minimumFractionDigits: d, maximumFractionDigits: d });
const sign = (v) => (Number(v) > 0 ? 'up' : Number(v) < 0 ? 'down' : 'dim');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 250); }, 4200);
}
const fail = (e) => toast(e.message || String(e), 'err');

/* ----------------------------------------------------------------- charts */
/* Charts are drawn by hand on a canvas: no chart library, so the page works
   with zero network access and nothing to keep patched. */
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const h = canvas.getAttribute('height') * 1 || 240;
  canvas.width = Math.max(1, rect.width * dpr);
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, h);
  return { ctx, w: rect.width, h };
}

/** Draw one or more series as lines with axes and a hover crosshair. */
function lineChart(canvas, labels, series, opts = {}) {
  const { ctx, w, h } = prepCanvas(canvas);
  if (!labels || labels.length < 2) return;
  const pad = { t: 12, r: 56, b: 24, l: 8 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;

  const all = series.flatMap((s) => s.data).filter((v) => Number.isFinite(v));
  if (!all.length) return;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;

  const x = (i) => pad.l + (i / (labels.length - 1)) * plotW;
  const y = (v) => pad.t + plotH - ((v - lo) / (hi - lo)) * plotH;

  // grid + right-hand axis labels
  ctx.strokeStyle = css('--border'); ctx.fillStyle = css('--text-faint');
  ctx.lineWidth = 1; ctx.font = '10px ' + css('--mono'); ctx.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, yy = y(v);
    ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + plotW, yy); ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillText(opts.fmt ? opts.fmt(v) : num(v, 0), pad.l + plotW + 7, yy + 3);
  }
  // x labels: first, middle, last
  ctx.textAlign = 'center';
  [0, Math.floor(labels.length / 2), labels.length - 1].forEach((i) => {
    // Clamp by half the label width, or the first/last date gets clipped by
    // the plot edge (a 10-char date at 10px mono is ~56px wide).
    const half = ctx.measureText(labels[i]).width / 2 + 2;
    ctx.fillText(labels[i], Math.min(Math.max(x(i), half), pad.l + plotW - half), h - 7);
  });

  series.forEach((s) => {
    ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.8;
    if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    let started = false;
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      started ? ctx.lineTo(x(i), y(v)) : (ctx.moveTo(x(i), y(v)), (started = true));
    });
    ctx.stroke(); ctx.setLineDash([]);
    if (s.fill) {
      ctx.lineTo(x(s.data.length - 1), y(lo)); ctx.lineTo(x(0), y(lo)); ctx.closePath();
      const g = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
      g.addColorStop(0, s.fill); g.addColorStop(1, 'transparent');
      ctx.fillStyle = g; ctx.fill();
    }
  });

  // crosshair + readout
  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.round(((ev.clientX - rect.left - pad.l) / plotW) * (labels.length - 1));
    if (i < 0 || i >= labels.length) return;
    lineChart(canvas, labels, series, { ...opts, _skipHover: true });
    const c2 = canvas.getContext('2d');
    c2.strokeStyle = css('--text-faint'); c2.lineWidth = 1; c2.setLineDash([3, 3]);
    c2.beginPath(); c2.moveTo(x(i), pad.t); c2.lineTo(x(i), pad.t + plotH); c2.stroke(); c2.setLineDash([]);
    series.forEach((s) => {
      if (!Number.isFinite(s.data[i])) return;
      c2.fillStyle = s.color; c2.beginPath(); c2.arc(x(i), y(s.data[i]), 3.2, 0, Math.PI * 2); c2.fill();
    });
    const txt = `${labels[i]}  ` + series.map((s) =>
      `${s.label} ${opts.fmt ? opts.fmt(s.data[i]) : num(s.data[i])}`).join('   ');
    c2.font = '11px ' + css('--sans'); c2.textAlign = 'left';
    const tw = c2.measureText(txt).width + 14;
    const tx = Math.min(x(i) + 8, w - tw - 4);
    c2.fillStyle = css('--bg-elev-2'); c2.strokeStyle = css('--border');
    c2.beginPath(); c2.roundRect(tx, pad.t + 2, tw, 22, 6); c2.fill(); c2.stroke();
    c2.fillStyle = css('--text'); c2.fillText(txt, tx + 7, pad.t + 17);
  };
  if (!opts._skipHover) canvas.onmouseleave = () => lineChart(canvas, labels, series, opts);
}

/** Horizontal allocation bars. */
function allocChart(canvas, items) {
  const { ctx, w, h } = prepCanvas(canvas);
  if (!items.length) {
    ctx.fillStyle = css('--text-faint'); ctx.font = '13px ' + css('--sans'); ctx.textAlign = 'center';
    ctx.fillText('No holdings yet', w / 2, h / 2);
    return;
  }
  const palette = ['#4f8cff', '#26c281', '#f0a92c', '#8b5cf6', '#f2555a', '#12b5cb', '#ec4899'];
  const total = items.reduce((a, b) => a + b.value, 0) || 1;
  const barH = Math.min(30, (h - 12) / items.length - 8);
  items.forEach((it, i) => {
    const yy = 8 + i * (barH + 8);
    const bw = Math.max(2, ((it.value / total) * (w - 130)));
    ctx.fillStyle = palette[i % palette.length];
    ctx.beginPath(); ctx.roundRect(84, yy, bw, barH, 5); ctx.fill();
    ctx.fillStyle = css('--text'); ctx.font = '600 12px ' + css('--sans'); ctx.textAlign = 'right';
    ctx.fillText(it.label, 76, yy + barH / 2 + 4);
    ctx.fillStyle = css('--text-dim'); ctx.font = '11px ' + css('--mono'); ctx.textAlign = 'left';
    ctx.fillText(`${((it.value / total) * 100).toFixed(1)}%`, 84 + bw + 7, yy + barH / 2 + 4);
  });
}

/* ------------------------------------------------------------------ tables */
function table(cols, rows, emptyMsg) {
  if (!rows.length) return `<div class="empty">${esc(emptyMsg)}</div>`;
  const head = cols.map((c) => `<th class="${c.num ? 'num' : ''}">${esc(c.label)}</th>`).join('');
  const body = rows.map((r) =>
    `<tr>${cols.map((c) => `<td class="${c.num ? 'num' : ''}">${c.render(r)}</td>`).join('')}</tr>`).join('');
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
const tag = (s) => `<span class="tag tag-${esc(s)}">${esc(s)}</span>`;

/* --------------------------------------------------------------- rendering */
function renderDashboard() {
  const p = state.portfolio;
  if (!p) return;
  document.getElementById('stat-cards').innerHTML = [
    { label: 'Total equity', value: money(p.equity), sub: `${money(p.cash)} cash available` },
    { label: 'Total P&L', value: money(p.total_pnl), sub: pct(p.total_pnl_pct), cls: sign(p.total_pnl) },
    { label: 'Positions value', value: money(p.positions_value), sub: `${p.positions.length} holding${p.positions.length === 1 ? '' : 's'}` },
    { label: 'Net deposits', value: money(p.net_deposits), sub: `${money(p.fees_paid)} fees paid` },
  ].map((c) => `<div class="card">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value ${c.cls || ''}">${c.value}</div>
      <div class="stat-sub ${c.cls || ''}">${c.sub}</div>
    </div>`).join('');

  const cols = [
    { label: 'Symbol', render: (r) => `<b>${esc(r.symbol)}</b>` },
    { label: 'Qty', num: true, render: (r) => num(r.quantity, 4) },
    { label: 'Avg cost', num: true, render: (r) => money(r.avg_price) },
    { label: 'Last', num: true, render: (r) => money(r.last_price) },
    { label: 'Value', num: true, render: (r) => money(r.market_value) },
    { label: 'P&L', num: true, render: (r) => `<span class="${sign(r.unrealised)}">${money(r.unrealised)}<br><small>${pct(r.unrealised_pct)}</small></span>` },
  ];
  document.getElementById('positions-table').innerHTML =
    table(cols, p.positions, 'No holdings. Fund the account, then place an order.');
  document.getElementById('trade-positions').innerHTML =
    table(cols, p.positions, 'No holdings yet.');

  const items = p.positions.map((x) => ({ label: x.symbol, value: x.market_value }));
  if (p.cash > 0) items.push({ label: 'Cash', value: p.cash });
  allocChart(document.getElementById('alloc-chart'), items);
  document.getElementById('buying-power').textContent = `Buying power: ${money(p.cash)}`;
}

function renderAccounts() {
  const sel = document.getElementById('f-account');
  sel.innerHTML = state.accounts.length
    ? state.accounts.map((a) => `<option value="${esc(a.id)}">${esc(a.nickname)} — ${esc(a.institution)} ${esc(a.masked)}</option>`).join('')
    : '<option value="">No linked accounts</option>';

  document.getElementById('accounts-list').innerHTML = state.accounts.length
    ? state.accounts.map((a) => `<div style="display:flex;justify-content:space-between;align-items:center;padding:11px 0;border-bottom:1px solid var(--border)">
        <div><b>${esc(a.nickname)}</b> ${a.is_default ? '<span class="tag tag-settled">default</span>' : ''}
          <div class="dim" style="font-size:12px">${esc(a.institution)} · ${esc(a.account_type)} · ${esc(a.masked)}</div></div>
        <button class="btn-sm btn-ghost" data-remove="${esc(a.id)}">Remove</button>
      </div>`).join('')
    : '<div class="empty">No funding sources linked yet.</div>';

  document.querySelectorAll('[data-remove]').forEach((b) => {
    b.onclick = async () => {
      try { await api.del(`/api/funding/accounts/${b.dataset.remove}`); toast('Account removed', 'ok'); await refresh(); }
      catch (e) { fail(e); }
    };
  });
}

function renderTransfers(rows) {
  const cols = [
    { label: 'Date', render: (r) => esc((r.created_at || '').slice(0, 10)) },
    { label: 'Type', render: (r) => esc(r.kind) },
    { label: 'Account', render: (r) => esc(r.nickname || '—') },
    { label: 'Amount', num: true, render: (r) => money(r.amount) },
    { label: 'Status', render: (r) => tag(r.status) },
  ];
  const html = table(cols, rows, 'No transfers yet.');
  document.getElementById('transfers-table').innerHTML = html;
  document.getElementById('activity-transfers').innerHTML = html;
}

function renderOrders(rows) {
  document.getElementById('orders-table').innerHTML = table([
    { label: 'Date', render: (r) => esc((r.created_at || '').slice(0, 10)) },
    { label: 'Symbol', render: (r) => `<b>${esc(r.symbol)}</b>` },
    { label: 'Side', render: (r) => tag(r.side) },
    { label: 'Type', render: (r) => esc(r.order_type) },
    { label: 'Qty', num: true, render: (r) => num(r.quantity, 4) },
    { label: 'Price', num: true, render: (r) => (r.filled_price ? money(r.filled_price) : '—') },
    { label: 'Fee', num: true, render: (r) => money(r.fee) },
    { label: 'Status', render: (r) => tag(r.status) + (r.reason ? `<div class="dim" style="font-size:11px">${esc(r.reason)}</div>` : '') },
  ], rows, 'No orders yet.');
}

/* -------------------------------------------------------------- backtesting */
function renderStrategyForm() {
  const s = state.strategies.find((x) => x.name === document.getElementById('b-strategy').value);
  if (!s) return;
  document.getElementById('strategy-desc').textContent = s.description;
  document.getElementById('symbol-b-field').hidden = !s.needs_pair;
  document.getElementById('strategy-params').innerHTML = s.params.map((p) => {
    if (p.kind === 'choice') {
      return `<div class="field"><label>${esc(p.label)}</label>
        <select data-param="${esc(p.name)}">${p.choices.map((c) =>
          `<option value="${esc(c)}" ${String(p.default) === c ? 'selected' : ''}>${c === '1' ? 'Yes' : c === '0' ? 'No' : esc(c)}</option>`).join('')}</select>
        ${p.help ? `<p class="hint">${esc(p.help)}</p>` : ''}</div>`;
    }
    return `<div class="field"><label>${esc(p.label)}</label>
      <input type="number" data-param="${esc(p.name)}" value="${esc(p.default)}"
        ${p.min != null ? `min="${p.min}"` : ''} ${p.max != null ? `max="${p.max}"` : ''}
        step="${p.step || (p.kind === 'float' ? 0.1 : 1)}">
      ${p.help ? `<p class="hint">${esc(p.help)}</p>` : ''}</div>`;
  }).join('');
}

function renderBacktest(r) {
  const s = r.stats, b = r.benchmark_stats;
  const metric = (label, value, cls = '') =>
    `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value ${cls}">${value}</div></div>`;

  document.getElementById('backtest-results').innerHTML = `
    <div class="card" style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
        <h2 style="margin:0">${esc(r.strategy)} · ${esc(r.symbol)}</h2>
        <span class="dim" style="font-size:12px">${esc(s.start)} → ${esc(s.end)} · ${s.days} bars</span>
      </div>
      <div class="metrics" style="margin-top:14px">
        ${metric('Total return', pct(s.total_return), sign(s.total_return))}
        ${metric('CAGR', pct(s.cagr), sign(s.cagr))}
        ${metric('Sharpe', num(s.sharpe), sign(s.sharpe))}
        ${metric('Sortino', num(s.sortino), sign(s.sortino))}
        ${metric('Max drawdown', pct(s.max_drawdown), 'down')}
        ${metric('Calmar', num(s.calmar), sign(s.calmar))}
        ${metric('Volatility', pct(s.volatility))}
        ${metric('Hit rate', pct(s.hit_rate))}
        ${metric('Profit factor', num(s.profit_factor))}
        ${metric('Exposure', pct(s.exposure))}
        ${metric('Trades', s.trades)}
        ${metric('Final equity', money(s.final_equity))}
      </div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <h2>Equity curve</h2>
      <div class="chart-wrap"><canvas id="equity-chart" height="250"></canvas></div>
      <div class="legend">
        <span><i class="swatch" style="background:${css('--accent')}"></i>Strategy ${pct(s.total_return)}</span>
        <span><i class="swatch" style="background:${css('--text-faint')}"></i>Buy &amp; hold ${pct(b.total_return)}</span>
      </div>
    </div>
    <div class="card" style="margin-bottom:14px">
      <h2>Drawdown</h2>
      <div class="chart-wrap"><canvas id="dd-chart" height="150"></canvas></div>
    </div>
    <div class="card">
      <h2>Trades <span class="dim" style="font-weight:400">(${r.trades.length}${r.trades.length > 100 ? ', showing first 100' : ''})</span></h2>
      <div style="max-height:420px;overflow-y:auto">
      ${table([
        { label: 'Entry', render: (t) => esc(t.entry_date) },
        { label: 'Exit', render: (t) => esc(t.exit_date) },
        { label: 'Side', render: (t) => tag(t.direction === 'long' ? 'buy' : 'sell') },
        { label: 'In', num: true, render: (t) => money(t.entry_price) },
        { label: 'Out', num: true, render: (t) => money(t.exit_price) },
        { label: 'P&L', num: true, render: (t) => `<span class="${sign(t.pnl)}">${money(t.pnl)}</span>` },
        { label: 'Return', num: true, render: (t) => `<span class="${sign(t.return_pct)}">${num(t.return_pct)}%</span>` },
        { label: 'Bars', num: true, render: (t) => t.bars_held },
        { label: 'Exit why', render: (t) => `<span class="dim">${esc(t.exit_reason)}</span>` },
      ], r.trades.slice(0, 100), 'No trades were taken — try loosening the filters.')}
      </div>
    </div>`;

  lineChart(document.getElementById('equity-chart'), r.dates, [
    { label: 'Strategy', data: r.equity, color: css('--accent'), width: 2, fill: 'rgba(79,140,255,.14)' },
    { label: 'B&H', data: r.benchmark, color: css('--text-faint'), width: 1.3, dash: [4, 4] },
  ], { fmt: (v) => '$' + (v / 1000).toFixed(0) + 'k' });

  lineChart(document.getElementById('dd-chart'), r.dates, [
    { label: 'Drawdown', data: r.drawdown.map((v) => v * 100), color: css('--down'), width: 1.5, fill: 'rgba(242,85,90,.16)' },
  ], { fmt: (v) => v.toFixed(0) + '%' });
}

/* ------------------------------------------------------------------ actions */
async function refresh() {
  try {
    const [pf, accts, trs, ords] = await Promise.all([
      api.get('/api/portfolio'), api.get('/api/funding/accounts'),
      api.get('/api/funding/transfers?limit=50'), api.get('/api/orders?limit=50'),
    ]);
    state.portfolio = pf; state.accounts = accts.accounts;
    renderDashboard(); renderAccounts(); renderTransfers(trs.transfers); renderOrders(ords.orders);
  } catch (e) { fail(e); }
}

async function loadQuote() {
  const sym = document.getElementById('t-symbol').value.trim().toUpperCase();
  if (!sym) return;
  try {
    const [q, h] = await Promise.all([api.get(`/api/quote/${sym}`), api.get(`/api/history/${sym}?days=180`)]);
    state.quote = q;
    document.getElementById('quote-price').textContent = money(q.price);
    document.getElementById('quote-change').innerHTML =
      `<span class="${sign(q.change)}">${q.change >= 0 ? '+' : ''}${num(q.change)} (${q.change_pct >= 0 ? '+' : ''}${num(q.change_pct)}%)</span>
       <span class="dim"> · as of ${esc(q.as_of)}${q.synthetic ? ' · simulated data' : ''}</span>`;
    document.getElementById('chart-title').textContent = `${sym} — last 180 sessions`;
    lineChart(document.getElementById('price-chart'), h.dates, [
      { label: sym, data: h.close, color: css('--accent'), width: 1.8, fill: 'rgba(79,140,255,.13)' },
    ], { fmt: (v) => '$' + v.toFixed(0) });
    updateEstimate();
  } catch (e) {
    document.getElementById('quote-price').textContent = '—';
    document.getElementById('quote-change').innerHTML = `<span class="down">${esc(e.message)}</span>`;
  }
}

function updateEstimate() {
  const qty = parseFloat(document.getElementById('t-qty').value) || 0;
  const price = state.quote ? state.quote.price : 0;
  const notional = qty * price, fee = notional * 0.0005;
  document.getElementById('est-notional').textContent = money(notional);
  document.getElementById('est-fee').textContent = money(fee);
  document.getElementById('est-total').textContent =
    money(state.side === 'buy' ? notional + fee : notional - fee);
  const btn = document.getElementById('submit-order');
  const sym = document.getElementById('t-symbol').value.trim().toUpperCase() || '—';
  btn.textContent = `${state.side === 'buy' ? 'Buy' : 'Sell'} ${sym}`;
  btn.className = `full ${state.side === 'buy' ? 'btn-buy' : 'btn-sell'}`;
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll('.nav-item[data-view]').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  ['dashboard', 'fund', 'trade', 'research', 'activity'].forEach((v) => {
    document.getElementById(`view-${v}`).hidden = v !== view;
  });
  if (view === 'trade') loadQuote();
  if (view === 'dashboard' && state.portfolio) renderDashboard();
}

/* -------------------------------------------------------------------- init */
function bind() {
  document.querySelectorAll('.nav-item[data-view]').forEach((b) =>
    (b.onclick = () => switchView(b.dataset.view)));

  document.getElementById('theme-toggle').onclick = () => {
    const light = document.documentElement.classList.toggle('light');
    localStorage.setItem('quantlab-theme', light ? 'light' : 'dark');
    document.getElementById('theme-label').textContent = light ? 'Dark mode' : 'Light mode';
    if (state.backtest) renderBacktest(state.backtest);
    if (state.portfolio) renderDashboard();
    if (state.view === 'trade') loadQuote();
  };

  document.getElementById('refresh-btn').onclick = refresh;

  document.getElementById('link-btn').onclick = async (ev) => {
    const body = {
      nickname: document.getElementById('f-nickname').value.trim(),
      institution: document.getElementById('f-institution').value.trim(),
      account_type: document.getElementById('f-type').value,
      last4: document.getElementById('f-last4').value.trim(),
    };
    if (!body.nickname || !body.institution || !body.last4) return toast('Fill in every field', 'err');
    ev.target.disabled = true;
    try {
      await api.post('/api/funding/accounts', body);
      toast('Funding source linked', 'ok');
      ['f-nickname', 'f-institution', 'f-last4'].forEach((id) => (document.getElementById(id).value = ''));
      await refresh();
    } catch (e) { fail(e); } finally { ev.target.disabled = false; }
  };

  const transfer = (kind) => async (ev) => {
    const amount = parseFloat(document.getElementById('f-amount').value);
    const account_id = document.getElementById('f-account').value || null;
    if (!(amount > 0)) return toast('Enter an amount', 'err');
    if (!account_id) return toast('Link a funding source first', 'err');
    ev.target.disabled = true;
    try {
      const r = await api.post(`/api/funding/${kind}`, { amount, account_id });
      toast(`${kind === 'deposit' ? 'Deposit' : 'Withdrawal'} ${r.status} — ${money(amount)}`, 'ok');
      await refresh();
    } catch (e) { fail(e); } finally { ev.target.disabled = false; }
  };
  document.getElementById('deposit-btn').onclick = transfer('deposit');
  document.getElementById('withdraw-btn').onclick = transfer('withdraw');

  document.querySelectorAll('#side-seg button').forEach((b) => {
    b.onclick = () => {
      state.side = b.dataset.side;
      document.querySelectorAll('#side-seg button').forEach((x) => {
        x.classList.toggle('active', x === b);
        x.classList.toggle('buy', x === b && state.side === 'buy');
        x.classList.toggle('sell', x === b && state.side === 'sell');
      });
      updateEstimate();
    };
  });

  let quoteTimer;
  document.getElementById('t-symbol').oninput = () => {
    clearTimeout(quoteTimer); quoteTimer = setTimeout(loadQuote, 450);
  };
  document.getElementById('t-qty').oninput = updateEstimate;
  document.getElementById('t-type').onchange = (e) => {
    document.getElementById('limit-field').hidden = e.target.value !== 'limit';
  };

  document.getElementById('submit-order').onclick = async (ev) => {
    const body = {
      symbol: document.getElementById('t-symbol').value.trim().toUpperCase(),
      side: state.side,
      quantity: parseFloat(document.getElementById('t-qty').value),
      order_type: document.getElementById('t-type').value,
    };
    if (body.order_type === 'limit') body.limit_price = parseFloat(document.getElementById('t-limit').value);
    if (!body.symbol || !(body.quantity > 0)) return toast('Check the symbol and quantity', 'err');
    ev.target.disabled = true;
    try {
      const o = await api.post('/api/orders', body);
      o.status === 'filled'
        ? toast(`${o.side} ${num(o.quantity, 4)} ${o.symbol} @ ${money(o.filled_price)}`, 'ok')
        : toast(`Order ${o.status}: ${o.reason || ''}`, 'err');
      await refresh();
    } catch (e) { fail(e); } finally { ev.target.disabled = false; updateEstimate(); }
  };

  document.getElementById('b-strategy').onchange = renderStrategyForm;

  document.getElementById('run-backtest').onclick = async (ev) => {
    const params = {};
    document.querySelectorAll('#strategy-params [data-param]').forEach((el) => {
      params[el.dataset.param] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
    });
    const body = {
      strategy: document.getElementById('b-strategy').value,
      symbol: document.getElementById('b-symbol').value.trim().toUpperCase(),
      symbol_b: document.getElementById('b-symbol-b').value.trim().toUpperCase() || null,
      start: document.getElementById('b-start').value || null,
      end: document.getElementById('b-end').value || null,
      capital: parseFloat(document.getElementById('b-capital').value) || 100000,
      target_volatility: parseFloat(document.getElementById('b-vol').value) || null,
      commission_bps: parseFloat(document.getElementById('b-commission').value) || 0,
      slippage_bps: parseFloat(document.getElementById('b-slippage').value) || 0,
      stop_loss_atr: parseFloat(document.getElementById('b-stop').value) || null,
      params,
    };
    ev.target.disabled = true;
    ev.target.innerHTML = '<span class="spinner"></span> Running…';
    try {
      const r = await api.post('/api/backtest', body);
      state.backtest = r; renderBacktest(r);
    } catch (e) {
      fail(e);
      document.getElementById('backtest-results').innerHTML =
        `<div class="card"><div class="empty">${esc(e.message)}</div></div>`;
    } finally { ev.target.disabled = false; ev.target.textContent = 'Run backtest'; }
  };

  document.getElementById('integrity-btn').onclick = async () => {
    try {
      const r = await api.get('/api/integrity');
      document.getElementById('integrity-result').innerHTML =
        `<div class="notice ${r.ok ? 'notice-info' : 'notice-warn'}"><span>${r.ok ? '✓' : '⚠︎'}</span><div>
          ${r.ok ? '<b>Ledger balances.</b>' : '<b>Ledger mismatch.</b>'}
          Cash re-derived from the transaction log is ${money(r.expected_cash)}; the recorded balance is
          ${money(r.actual_cash)}. Difference ${money(r.difference)}.</div></div>`;
    } catch (e) { fail(e); }
  };

  document.getElementById('reset-btn').onclick = async () => {
    if (!confirm('Delete every position, order, transfer and linked account in the paper account?')) return;
    try { await api.post('/api/admin/reset', { confirm: true }); toast('Account reset', 'ok'); await refresh(); }
    catch (e) { fail(e); }
  };

  window.addEventListener('resize', () => {
    if (state.backtest && state.view === 'research') renderBacktest(state.backtest);
    if (state.view === 'dashboard' && state.portfolio) renderDashboard();
  });
}

async function init() {
  if (localStorage.getItem('quantlab-theme') === 'light') {
    document.documentElement.classList.add('light');
    document.getElementById('theme-label').textContent = 'Dark mode';
  }
  document.getElementById('b-end').value = new Date().toISOString().slice(0, 10);
  bind();
  try {
    const s = await api.get('/api/strategies');
    state.strategies = s.strategies;
    document.getElementById('b-strategy').innerHTML = s.strategies
      .map((x) => `<option value="${esc(x.name)}">${esc(x.label)} · ${esc(x.category)}</option>`).join('');
    renderStrategyForm();
  } catch (e) { fail(e); }
  await refresh();
}

document.addEventListener('DOMContentLoaded', init);
