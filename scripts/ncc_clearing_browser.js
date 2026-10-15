// Run in DevTools Snippets on the listed NCC pages after expanding archive sections.
// Captures visible document links and tables, never cookies, tokens or form inputs.
(async () => {
  const paths = ["/catalog/020415/93", "/catalog/0204/248", "/rates/settleDates"];
  if (location.origin !== "https://www.nationalclearingcentre.ru" || !paths.includes(location.pathname)) {
    throw new Error("Open NCC clearing rules, decisions or settlement calendar first");
  }
  const visible = element => element.getClientRects().length > 0 && getComputedStyle(element).visibility !== "hidden";
  const hex = bytes => [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, "0")).join("");
  const jobs = [...new Map([...document.querySelectorAll("a[href]")].filter(a => {
    const u = new URL(a.href);
    return visible(a) && u.origin === location.origin && u.pathname === "/connector" && u.searchParams.get("cmd") === "file";
  }).map(a => [a.href, a])).values()];
  const result = {schema_version: 1, kind: "ncc_clearing_browser_capture", page_url: location.href,
    captured_at: new Date().toISOString(), visible_document_count: jobs.length, documents: [],
    tables: [...document.querySelectorAll("table")].filter(visible).map(t => t.innerText),
    completeness_verified: false};
  if (!jobs.length && !result.tables.length) throw new Error("No visible documents or calendar tables. Open archive/table in the page UI first.");
  let total = 0;
  for (const a of jobs) {
    let parent = a.parentElement;
    for (let i = 0; i < 3 && parent.parentElement && parent.innerText.length < 200; i++) parent = parent.parentElement;
    const row = {url: a.href, title: a.textContent.trim(), context: parent.innerText.slice(0, 16000)};
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
  console.log("Save JSON to F:/moex-data/specifications/incoming. This captures only currently visible documents/tables.");
})();
