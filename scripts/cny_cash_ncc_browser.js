// Run on https://www.nationalclearingcentre.ru/catalog/0204/106
(async () => {
  if (location.origin !== 'https://www.nationalclearingcentre.ru') throw new Error('Open the NCC tariff archive');
  const requests = [
  {
    "name": "ncc-march.pdf",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1773754623&cmd=file&target=B_XENsZWFyaW5nXNCU0L7QutGD0LzQtdC90YLRi1zQotCw0YDQuNGE0Ysg0LrQu9C40YDQuNC90LPQvtCy0L7Qs9C_PINGG0LXQvdGC0YDQsFzQndCa0KYg0KLQsNGA0LjRhNGLICgxLdCw0Y8g0YDQtdC00LDQutGG0LjRjyAyMDI2KV_SQsdC10Lcg0L_SRgNCw0LLQvtC6LnBkZg_E_E"
  },
  {
    "name": "ncc-june.pdf",
    "url": "https://www.nationalclearingcentre.ru/connector?cmd=file&target=B_XENsZWFyaW5nXNCU0L7QutGD0LzQtdC90YLRi1zQotCw0YDQuNGE0Ysg0LrQu9C40YDQuNC90LPQvtCy0L7Qs9C_PINGG0LXQvdGC0YDQsFzQndCa0KYg0KLQsNGA0LjRhNGLICgyLdCw0Y8g0YDQtdC00LDQutGG0LjRjyAyMDI2KV_SQsdC10Lcg0L_SRgNCw0LLQvtC6LnBkZg_E_E"
  },
  {
    "name": "ncc-july.pdf",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1782297457&cmd=file&target=A_XNCS0LvQsNC00LjQvNC40YDQvtCy0LBc0J3QmtCmINCi0LDRgNC40YTRiyAoMy3QsNGPINGA0LXQtNCw0LrRhtC40Y8gMjAyNilf0LHQtdC3INC_S0YDQsNCy0L7Qui5wZGY_E"
  }
];
  const output = {schema_version: 1, page_url: location.href, observed_at: new Date().toISOString(), files: []};
  for (const item of requests) {
    const row = {...item};
    try {
      const response = await fetch(item.url, {credentials: 'same-origin'});
      row.status = response.status;
      row.final_url = response.url;
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (new TextDecoder().decode(bytes.slice(0, 5)) !== '%PDF-') throw new Error('Response is not a PDF');
      row.bytes = bytes.length;
      row.sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
      let binary = '';
      for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768));
      row.base64 = btoa(binary);
    } catch (error) { row.error = String(error); }
    output.files.push(row);
    console.log(item.name, row.error || ('PDF: ' + row.bytes + ' bytes'));
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(output, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a');
  a.href = url; a.download = 'cny-cash-ncc-' + Date.now() + '.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
})();
