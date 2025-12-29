(()=>{
// Remove contents of any previous tracerite elements based on cleanup mode
const current=document.currentScript?.parentElement;
if(current?.dataset.replacePrevious){
  const mode=current.dataset.cleanupMode||'replace';
  document.querySelectorAll('.tracerite').forEach(el=>{
    if(el!==current){
      // Always remove script and style to prevent conflicts
      el.querySelectorAll('script,style').forEach(c=>c.remove());
      // In replace mode, also remove the traceback content (keep h2)
      if(mode==='replace')el.querySelectorAll('.traceback-details,.traceback-ellipsis,h3,.excmessage').forEach(c=>c.remove());
    }
  });
}
})()
