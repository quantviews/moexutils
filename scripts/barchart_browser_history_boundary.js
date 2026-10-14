// Run as a DevTools Snippet on https://www.barchart.com while signed in.
// 5 sequential, same-origin requests; set START_INDEX to resume. No cookies or auth headers are read/exported.
// Captures raw responses only; this is NOT a validated market-data package.
(async () => {
  if (location.origin !== "https://www.barchart.com") {
    throw new Error("Run this snippet on https://www.barchart.com");
  }
  const START_INDEX = 0; // Zero-based index printed on a failure.
  const symbols = ["KCH25", "OJF25", "OJH25", "TGF25", "TGG25"];
  const capture = {
    schema_version: 2,
    campaign: "historical-2024-boundary-20261014",
    status: "unvalidated_browser_capture",
    captured_at: new Date().toISOString(),
    intended_comparison_start: "2024-10-01",
    intended_comparison_end: "2024-12-31",
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
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
  const blob = new Blob([JSON.stringify(capture, null, 2)], { type: "application/json" });
  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = "barchart-history-boundary-" + capture.captured_at.replace(/[:.]/g, "-") + ".json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);
  console.info("Saved raw responses:", capture.responses.length, "of", symbols.length,
    capture.stopped || "Check symbols, dates, frequency and price fields before use.");
})();
