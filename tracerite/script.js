(()=>{
// Remove contents of any previous tracerite elements, preserving wrapper and headers
const current=document.currentScript?.parentElement;
if(current?.dataset.replacePrevious)document.querySelectorAll('.tracerite').forEach(el=>{
  if(el!==current)el.querySelectorAll('script,style,.traceback-wrapper').forEach(c=>c.remove());
});
const scrollto=id=>document.getElementById(id).scrollIntoView({behavior:'smooth',block:'nearest',inline:'start'});
window.tracerite_scrollto=scrollto;

// Toggle modifier class for cursor styling
const updateModifier=e=>{current?.classList.toggle('modifier-active',e.ctrlKey||e.shiftKey||e.altKey||e.metaKey)};
addEventListener('keydown',updateModifier);
addEventListener('keyup',updateModifier);
})()
