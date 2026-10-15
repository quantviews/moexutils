// Run on the public NCC rules archive; downloads only nine outstanding originals.
(async () => {
  if (location.origin !== 'https://www.nationalclearingcentre.ru' || location.pathname !== '/catalog/020415/93')
    throw new Error('Open https://www.nationalclearingcentre.ru/catalog/020415/93');
  const requests = [
  {
    "approved_date": "2022-11-21",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1671523408&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVw0LiDQndCa0KYg0J_SRgNCw0LLQuNC70LAg0LrQu9C40YDQuNC90LPQsC4g0J3QsCDRgdGA0L7Rh9C90L7QvCDRgNGL0L3QutC1ICg0LdGPINGA0LXQtNCw0LrRhtC40Y8gMjAyMilf0LHQtdC3INC_S0YDQsNCy0L7Qui5wZGY_E"
  },
  {
    "approved_date": "2024-04-25",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1721829972&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVwzLiDQndCa0KYg0J_SRgNCw0LLQuNC70LAg0LrQu9C40YDQuNC90LPQsC4g0J3QsCDRgdGA0L7Rh9C90L7QvCDRgNGL0L3QutC1ICgzLdGPINGA0LXQtNCw0LrRhtC40Y8gMjAyNClf0LHQtdC3INC_S0YDQsNCy0L7Qui5wZGY_E"
  },
  {
    "approved_date": "2024-07-31",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1724936429&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVwzLiDQndCa0KYg0J_SRgNCw0LLQuNC70LAg0LrQu9C40YDQuNC90LPQsC4g0J3QsCDRgdGA0L7Rh9C90L7QvCDRgNGL0L3QutC1ICg0LdGPINGA0LXQtNCw0LrRhtC40Y8gMjAyNClf0YfQuNGB0YLRi9C5LnBkZg_E_E"
  },
  {
    "approved_date": "2024-10-31",
    "url": "https://www.nationalclearingcentre.ru/connector?cmd=file&target=B_XENsZWFyaW5nXNCU0L7QutGD0LzQtdC90YLRi1zQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwXDQuINCd0JrQpiDQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwLiDQndCwINGB0YDQvtGH0L3QvtC8INGA0YvQvdC60LUgKNC00LXQutCw0LHRgNGMIDIwMjQpLnBkZg_E_E"
  },
  {
    "approved_date": "2025-02-11",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1741867940&cmd=file&target=B_XENsZWFyaW5nXNCU0L7QutGD0LzQtdC90YLRi1zQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwXDQuINCd0JrQpiDQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwLiDQndCwINGB0YDQvtGH0L3QvtC8INGA0YvQvdC60LUgKDEt0Y8g0YDQtdC00LDQutGG0LjRjyAyMDI1KS5wZGY_E"
  },
  {
    "approved_date": "2023-03-01",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1680074395&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVw0LTEuINCd0JrQpiDQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwLiDQndCwINGB0YDQvtGH0L3QvtC8INGA0YvQvdC60LUgKDEt0Y8g0YDQtdC00LDQutGG0LjRjyAyMDIzKV_SQsdC10Lcg0L_SRgNCw0LLQvtC6XzIucGRm"
  },
  {
    "approved_date": "2023-06-23",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1689872068&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVwxLTMuINCd0JrQpiDQn9GA0LDQstC40LvQsCDQutC70LjRgNC40L3Qs9CwLiDQndCwINGB0YDQvtGH0L3QvtC8INGA0YvQvdC60LUgKDIt0Y8g0YDQtdC00LDQutGG0LjRjyAyMDIzKV_SQsdC10Lcg0L_SRgNCw0LLQvtC6LnBkZg_E_E"
  },
  {
    "approved_date": "2023-08-28",
    "url": "https://nationalclearingcentre.ru/connector?_t=1695148059&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVw0LiDQndCa0KYg0J_SRgNCw0LLQuNC70LAg0LrQu9C40YDQuNC90LPQsC4g0J3QsCDRgdGA0L7Rh9C90L7QvCDRgNGL0L3QutC1ICgzLdGPINGA0LXQtNCw0LrRhtC40Y8gMjAyMylf0LjQt9GK0Y_SRgtC40Y8ucGRm"
  },
  {
    "approved_date": "2025-04-24",
    "url": "https://www.nationalclearingcentre.ru/connector?_t=1750173339&cmd=file&target=B_XENsZWFyaW5nXNCf0YDQsNCy0LjQu9CwINC60LvQuNGA0LjQvdCz0LBc0KfQsNGB0YLRjCBJViDQodGA0L7Rh9C90YvQuVw0LiDQndCa0KYg0J_SRgNCw0LLQuNC70LAg0LrQu9C40YDQuNC90LPQsC4g0J3QsCDRgdGA0L7Rh9C90L7QvCDRgNGL0L3QutC1ICgyLdGPINGA0LXQtNCw0LrRhtC40Y8gMjAyNSkucGRm"
  }
];
  const result = {schema_version:1, kind:'ncc_clearing_browser_capture', page_url:location.href,
    captured_at:new Date().toISOString(), document_link_count:requests.length, documents:[], completeness_verified:false};
  const hex = bytes => [...new Uint8Array(bytes)].map(x=>x.toString(16).padStart(2,'0')).join('');
  for (const job of requests) {
    const url = new URL(job.url); url.hostname = location.hostname;
    const row = {url:url.href, title:job.approved_date, context:'Targeted outstanding original; applicability requires review'};
    try {
      const response = await fetch(url, {credentials:'same-origin',signal:AbortSignal.timeout(30000)});
      row.status=response.status; row.received_at=new Date().toISOString(); row.content_type=response.headers.get('content-type');
      if (!response.ok) throw new Error('HTTP '+response.status);
      const buffer=await response.arrayBuffer(); const bytes=new Uint8Array(buffer);
      if (bytes.length>30*1024*1024 || new TextDecoder().decode(bytes.subarray(0,5))!=='%PDF-')
        throw new Error('Not a PDF or larger than 30 MiB');
      row.sha256=hex(await crypto.subtle.digest('SHA-256',buffer));row.format='pdf';
      let binary='';for(let i=0;i<bytes.length;i+=8192) binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
      row.base64=btoa(binary);
    } catch(e) { row.error=String(e); }
    result.documents.push(row);
    console.log(job.approved_date,row.error || 'saved');
    if(row.error && ![404,410].includes(row.status)){result.stopped=row.error;break;}
    await new Promise(r=>setTimeout(r,1500));
  }
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(result)],{type:'application/json'}));
  a.download='ncc-remaining-'+Date.now()+'.json';
  a.textContent='Сохранить JSON НКЦ';a.style.cssText='position:fixed;top:10px;left:10px;z-index:2147483647;background:white;padding:20px';
  document.body.append(a);globalThis.nccCaptureResult=result;a.click();
})();
