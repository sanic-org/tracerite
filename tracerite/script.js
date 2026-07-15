(()=>{
// Move style to head (replacing any old version) to preserve it when .tracerite elements are deleted
const current=document.currentScript?.parentElement
const style=current?.querySelector('style')
if(style){
  document.getElementById('tracerite-style')?.remove()
  style.id='tracerite-style'
  document.head.appendChild(style)
}
// Remove contents of previous tracerite elements when in replace cleanup mode
if(current?.dataset.cleanupMode==='replace'){
  document.querySelectorAll('.tracerite').forEach(el=>{
    if(el!==current){
      // Remove all direct children except h2
      [...el.children].forEach(c=>{if(c.tagName!=='H2')c.remove()})
    }
  })
}
// Disable autodark when the effective background is light
if(current?.classList.contains('autodark')){
  const isTransparent=(bg)=>!bg || bg==='transparent' || /rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)/.test(bg)
  const getBg=(el)=>{
    while(el && el!==document.documentElement){
      const bg=getComputedStyle(el).backgroundColor
      if(!isTransparent(bg)) return bg
      el=el.parentElement
    }
    for(const node of [document.body, document.documentElement]){
      const bg=getComputedStyle(node).backgroundColor
      if(!isTransparent(bg)) return bg
    }
    return 'rgb(255,255,255)'
  }
  const red=getBg(current).match(/rgba?\(\s*(\d+)/)?.[1]
  if(red && parseInt(red)>=128){
    current.classList.remove('autodark')
  }
}
})()
