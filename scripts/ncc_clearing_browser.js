// Run in DevTools Snippets on the listed NCC pages after expanding archive sections.
// Captures document links present in DOM and visible tables; no cookies or form inputs.
(async () => {
  const paths = ["/catalog/020415/93", "/catalog/0204/248", "/rates/settleDates"];
  if (location.origin !== "https://www.nationalclearingcentre.ru" || !paths.includes(location.pathname)) {
    throw new Error("Open NCC clearing rules, decisions or settlement calendar first");
  }
  const visible = element => element.getClientRects().length > 0 && getComputedStyle(element).visibility !== "hidden";
  const hex = bytes => [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, "0")).join("");
  const jobs = [...new Map([...document.querySelectorAll("a[href]")].filter(a => {
    const u = new URL(a.href);
    return u.origin === location.origin && u.pathname === "/connector" && u.searchParams.get("cmd") === "file";
  }).map(a => [a.href, a])).values()];
  const result = {schema_version: 1, kind: "ncc_clearing_browser_capture", page_url: location.href,
    captured_at: new Date().toISOString(), visible_document_count: jobs.filter(visible).length,
    document_link_count: jobs.length, documents: [],
    tables: [...document.querySelectorAll("table")].filter(visible).map(t => t.innerText),
    completeness_verified: false};
  if (location.pathname === "/rates/settleDates") {
    // Calendar may use div grids, an iframe or XHR instead of a table.
    // Always save diagnostics, including when nothing has rendered yet.
    const safeUrl = value => {
      try {
        const u = new URL(value, location.href);
        if (u.protocol !== "https:" || u.username || u.password ||
            [...u.searchParams.keys()].some(k => /token|auth|session|key|password|email/i.test(k))) return null;
        u.hash = "";
        return u.href;
      } catch { return null; }
    };
    const root = document.querySelector("main, [role=main], article") || document.body;
    const clone = root.cloneNode(true);
    clone.querySelectorAll("form,input,textarea,select,script,style,nav,header,footer").forEach(e => e.remove());
    result.calendar_diagnostics = {
      page_text: clone.textContent.replace(/\s+/g, " ").trim().slice(0, 120000),
      grids: [...document.querySelectorAll('[role="grid"], [role="table"]')].filter(visible).map(e => e.innerText),
      frames: [...document.querySelectorAll("iframe[src]")].map(e => safeUrl(e.src)).filter(Boolean),
      scripts: [...document.querySelectorAll("script[src]")].map(e => safeUrl(e.src)).filter(Boolean),
      resource_urls: [], responses: []
    };
    const resources = performance.getEntriesByType("resource").filter(e =>
      ["fetch", "xmlhttprequest"].includes(e.initiatorType) && /settle|calendar|\/rates\//i.test(e.name));
    const urls = [...new Set(resources.map(e => safeUrl(e.name)).filter(Boolean))];
    result.calendar_diagnostics.resource_urls = urls;
    for (const url of urls.filter(u => new URL(u).origin === location.origin).slice(0, 10)) {
      const row = {url, observed_at: new Date().toISOString()};
      try {
        const response = await fetch(url, {credentials: "same-origin"});
        row.status = response.status;
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        row.content_type = response.headers.get("content-type");
        const body = await response.arrayBuffer();
        if (body.byteLength > 4 * 1024 * 1024) throw new Error("Response exceeds 4 MiB; not included");
        row.sha256 = hex(await crypto.subtle.digest("SHA-256", body));
        let binary = "";
        const bytes = new Uint8Array(body);
        for (let pos = 0; pos < bytes.length; pos += 8192) binary += String.fromCharCode(...bytes.subarray(pos, pos + 8192));
        row.base64 = btoa(binary);
      } catch (error) { row.error = String(error); }
      result.calendar_diagnostics.responses.push(row);
      if (row.status === 403 || row.status === 429) break;
    }
    result.calendar_status = result.tables.length || result.calendar_diagnostics.grids.length
      ? "visible_data_pending_review" : "diagnostic_capture_not_calendar_data";
  } else if (!jobs.length) {
    result.warning = "No document links in DOM; expand archive using the page UI and repeat";
  }
  let total = 0;
  for (const a of jobs) {
    let parent = a.parentElement;
    for (let i = 0; i < 3 && parent.parentElement && parent.innerText.length < 200; i++) parent = parent.parentElement;
    const row = {url: a.href, title: a.textContent.trim(), visible: visible(a),
      context: (parent.innerText || parent.textContent).slice(0, 16000), context_scope: "requires_manual_association"};
    try {
      const response = await fetch(a.href, {credentials: "same-origin"});
      row.status = response.status;
      row.received_at = new Date().toISOString();
      row.content_type = response.headers.get("content-type");
      if (!response.ok) throw new Error(`HTTP ${response.status}; no retry`);
      const buffer = await response.arrayBuffer();
      total += buffer.byteLength;
      if (total > 200 * 1024 * 1024) throw new Error("200 MiB batch limit; partial result saved");
      const bytes = new Uint8Array(buffer);
      const pdf = bytes[0] === 0x25 && bytes[1] === 0x50 && bytes[2] === 0x44 && bytes[3] === 0x46;
      const zip = bytes[0] === 0x50 && bytes[1] === 0x4b;
      const ole = bytes[0] === 0xd0 && bytes[1] === 0xcf && bytes[2] === 0x11 && bytes[3] === 0xe0;
      if (!(pdf || zip || ole)) throw new Error("Unexpected file signature; not accepting HTML");
      row.format = pdf ? "pdf" : zip ? "zip_container_pending_review" : "ole_container_pending_review";
      row.sha256 = hex(await crypto.subtle.digest("SHA-256", buffer));
      let binary = "";
      for (let pos = 0; pos < bytes.length; pos += 8192) binary += String.fromCharCode(...bytes.subarray(pos, pos + 8192));
      row.base64 = btoa(binary);
      result.documents.push(row);
      console.log(`Saved ${result.documents.length}/${jobs.length}`);
    } catch (error) {
      row.error = String(error);
      result.documents.push(row);
      result.stopped = row.error;
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(result)], {type: "application/json"}));
  const a = document.createElement("a");
  a.href = url;
  a.download = `ncc-clearing-${location.pathname.split("/").pop()}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  console.log("Save JSON to F:/moex-data/specifications/incoming. Document links present in DOM are included; calendar may contain diagnostics only.");
})();
