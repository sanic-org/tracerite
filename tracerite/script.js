(()=>{
// Remove contents of any previous tracerite elements, preserving wrapper and headers
const current=document.currentScript?.parentElement;
if(current?.dataset.replacePrevious)document.querySelectorAll('.tracerite').forEach(el=>{
  if(el!==current)el.querySelectorAll('script,style,.traceback-wrapper').forEach(c=>c.remove());
});
})()
