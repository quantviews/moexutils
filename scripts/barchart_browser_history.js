// Run as a DevTools Snippet on https://www.barchart.com while signed in.
// 115 sequential, same-origin requests; set START_INDEX to resume. No cookies or auth headers are read/exported.
// Captures raw responses only; this is NOT a validated market-data package.
(async () => {
  if (location.origin !== "https://www.barchart.com") {
    throw new Error("Run this snippet on https://www.barchart.com");
  }
  const START_INDEX = 0; // Zero-based index printed on a failure.
  const symbols = ["KCH20", "KCK20", "KCN20", "KCU20", "KCZ20", "OJF20", "OJH20", "OJK20", "OJN20", "OJU20", "OJX20", "TGF20", "TGG20", "TGH20", "TGJ20", "TGK20", "TGM20", "TGN20", "TGQ20", "TGU20", "TGV20", "TGX20", "TGZ20", "KCH21", "KCK21", "KCN21", "KCU21", "KCZ21", "OJF21", "OJH21", "OJK21", "OJN21", "OJU21", "OJX21", "TGF21", "TGG21", "TGH21", "TGJ21", "TGK21", "TGM21", "TGN21", "TGQ21", "TGU21", "TGV21", "TGX21", "TGZ21", "KCH22", "KCK22", "KCN22", "KCU22", "KCZ22", "OJF22", "OJH22", "OJK22", "OJN22", "OJU22", "OJX22", "TGF22", "TGG22", "TGH22", "TGJ22", "TGK22", "TGM22", "TGN22", "TGQ22", "TGU22", "TGV22", "TGX22", "TGZ22", "KCH23", "KCK23", "KCN23", "KCU23", "KCZ23", "OJF23", "OJH23", "OJK23", "OJN23", "OJU23", "OJX23", "TGF23", "TGG23", "TGH23", "TGJ23", "TGK23", "TGM23", "TGN23", "TGQ23", "TGU23", "TGV23", "TGX23", "TGZ23", "KCH24", "KCK24", "KCN24", "KCU24", "KCZ24", "OJF24", "OJH24", "OJK24", "OJN24", "OJU24", "OJX24", "TGF24", "TGG24", "TGH24", "TGJ24", "TGK24", "TGM24", "TGN24", "TGQ24", "TGU24", "TGV24", "TGX24", "TGZ24"];
  const capture = {
    schema_version: 2,
    campaign: "historical-2020-2024-full-20261009",
    status: "unvalidated_browser_capture",
    captured_at: new Date().toISOString(),
    intended_comparison_start: "2019-01-01",
    intended_comparison_end: "2024-11-01",
    // No assumed date-filter syntax for this undocumented website proxy.
    // Ask for 10000 records; the website may impose a lower cap. Verify after import.
    responses: [],
  };
  for (let index = START_INDEX; index < symbols.length; index++) {
    const symbol = symbols[index];
    const url = new URL("/proxies/timeseries/historical/queryeod.ashx", location.origin);
    url.search = new URLSearchParams({
      symbol, data: "daily", maxrecords: "10000", volume: "contract",
      order: "asc", dividends: "false", backadjust: "false",
      splits: "false", padded: "false",
    }).toString();
    try {
      const response = await fetch(url, {
        credentials: "same-origin", redirect: "error", cache: "no-store",
        signal: AbortSignal.timeout(30000), headers: { Accept: "*/*" },
      });
      const body = await response.text();
      const contentType = response.headers.get("content-type") || "";
      // Do not export login/challenge pages or response headers.
      if (!response.ok || !body.trim() || /html/i.test(contentType) || /^\s*</.test(body)) {
        capture.stopped = { index, symbol, http_status: response.status, reason: "HTTP error, empty response, or HTML/XML; inspect Network locally" };
        break;
      }
      if (!body.trimStart().startsWith(symbol + ",") || body.trim().split(/\r?\n/).some(line => line.split(",").length !== 8)) {
        capture.stopped = { index, symbol, reason: "Unexpected data format; inspect Network locally" };
        break;
      }
      capture.responses.push({ symbol, url: url.href, received_at: new Date().toISOString(),
        http_status: response.status, content_type: contentType, body });
      console.info(index, symbol, response.status, body.length, "characters; format not yet validated");
    } catch {
      capture.stopped = { index, symbol, reason: "Network error, redirect, or timeout; inspect Network locally" };
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  const blob = new Blob([JSON.stringify(capture, null, 2)], { type: "application/json" });
  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = "barchart-history-" + capture.captured_at.replace(/[:.]/g, "-") + ".json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);
  console.info("Saved raw responses:", capture.responses.length, "of", symbols.length,
    capture.stopped || "Check symbols, dates, frequency and price fields before use.");
})();
