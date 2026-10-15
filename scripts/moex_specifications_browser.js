// Run on https://www.moex.com/ru/documents/4979 in your browser.
// Repeat after selecting each archive page (1..4) in the site's normal UI.
// Downloads only links visible on the current page; no cookies/tokens exported.
(async () => {
  if (location.origin !== "https://www.moex.com" || !location.pathname.endsWith("/documents/4979")) {
    throw new Error("Open the MOEX document 4979 archive first");
  }
  const title = "Фьючерсные контракты на акции российских эмитентов";
  const links = [...document.querySelectorAll("a[href]")].filter(a => {
    const url = new URL(a.href, location.href);
    return a.textContent.includes(title) && url.origin === location.origin && url.pathname.startsWith("/files/");
  });
  const jobs = [...new Map(links.map(a => [a.href, a])).values()];
  if (!jobs.length) throw new Error("No specification links found on this page; nothing downloaded");
  const hex = bytes => [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, "0")).join("");
  const result = {schema_version: 1, kind: "moex_specification_browser_capture", page_url: location.href,
    captured_at: new Date().toISOString(), association_status: "page_context_requires_review", documents: []};
  for (const a of jobs) {
    let context = a.parentElement;
    for (let i = 0; i < 5 && context.parentElement && !/Период действия|Действует:|Вступает в силу/.test(context.innerText); i++) {
      context = context.parentElement;
    }
    const row = {url: a.href, title: a.textContent.trim(), context: context.innerText.slice(0, 12000)};
    try {
      const response = await fetch(a.href, {credentials: "same-origin"});
      row.status = response.status;
      row.received_at = new Date().toISOString();
      row.content_type = response.headers.get("content-type");
      if (!response.ok) throw new Error(`HTTP ${response.status}; stop without retry`);
      const buffer = await response.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      // DOCX (ZIP), old DOC (OLE), or PDF. Never accept HTML/login/error as a file.
      const zip = bytes[0] === 0x50 && bytes[1] === 0x4b;
      const ole = bytes[0] === 0xd0 && bytes[1] === 0xcf && bytes[2] === 0x11 && bytes[3] === 0xe0;
      const pdf = bytes[0] === 0x25 && bytes[1] === 0x50 && bytes[2] === 0x44 && bytes[3] === 0x46;
      if (!(zip || ole || pdf)) throw new Error("Unexpected document signature; stop");
      row.format = zip ? "docx" : ole ? "doc" : "pdf";
      row.sha256 = hex(await crypto.subtle.digest("SHA-256", buffer));
      let binary = "";
      for (let pos = 0; pos < bytes.length; pos += 8192) binary += String.fromCharCode(...bytes.subarray(pos, pos + 8192));
      row.base64 = btoa(binary);
      result.documents.push(row);
      console.log(`Saved ${result.documents.length}/${jobs.length}: ${a.href}`);
    } catch (error) {
      row.error = String(error);
      result.documents.push(row);
      result.stopped = row.error;
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  const blob = new Blob([JSON.stringify(result)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const download = document.createElement("a");
  download.href = url;
  download.download = `moex-specifications-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  download.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  console.log("Save this JSON. Select the next archive page in MOEX UI and run again.");
})();
