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

// Update floating symbols - position at edge pointing toward target, animate when visible
const updateFloatingSymbols=(wrapper,container)=>{
  const floaters=wrapper.querySelectorAll('.floating-symbol');
  const wrapperRect=wrapper.getBoundingClientRect();
  const containerRect=container.getBoundingClientRect();
  const pad=8;

  floaters.forEach(f=>{
    const frameId=f.dataset.frame;
    const frame=document.getElementById(frameId);
    if(!frame){f.classList.remove('visible');return;}

    const frameRect=frame.getBoundingClientRect();
    // Find the actual symbol within this frame
    const sym=frame.querySelector('.tracerite-symbol');
    const frameContent=frame.querySelector('.frame-content');

    // Target: the actual symbol position (accounting for scroll)
    let targetX,targetY;
    if(sym&&frameContent){
      const symRect=sym.getBoundingClientRect();
      targetX=symRect.left+symRect.width/2;
      targetY=symRect.top+symRect.height/2;
    }else{
      targetX=frameRect.left+frameRect.width/2;
      targetY=frameRect.top+Math.min(30,frameRect.height/2);
    }

    // Check horizontal visibility
    const visibleLeft=Math.max(frameRect.left,containerRect.left);
    const visibleRight=Math.min(frameRect.right,containerRect.right);
    const visibleWidth=Math.max(0,visibleRight-visibleLeft);
    const isPartiallyVisible=visibleWidth>10;

    // Check if target point itself is visible
    const targetInView=targetX>=containerRect.left&&targetX<=containerRect.right&&
                       targetY>=wrapperRect.top&&targetY<=wrapperRect.bottom;

    if(targetInView){
      // Animate to target then fade out
      const x=targetX-wrapperRect.left;
      const y=targetY-wrapperRect.top;
      f.classList.add('animating','fading');
      f.style.left=x+'px';
      f.style.top=y+'px';
      setTimeout(()=>{
        f.classList.remove('visible','animating','fading');
      },250);
      return;
    }

    if(isPartiallyVisible&&f.classList.contains('visible')){
      // Frame partially visible but symbol not in view - animate toward symbol
      const x=Math.max(pad,Math.min(wrapperRect.width-pad,targetX-wrapperRect.left));
      const y=Math.max(pad,Math.min(wrapperRect.height-pad,targetY-wrapperRect.top));
      f.style.left=x+'px';
      f.style.top=y+'px';
      return;
    }

    // Not visible - position at edge pointing to target
    const cx=wrapperRect.left+wrapperRect.width/2;
    const cy=wrapperRect.top+wrapperRect.height/2;
    const dx=targetX-cx;
    const dy=targetY-cy;

    let x,y;
    const w=wrapperRect.width-pad*2;
    const h=wrapperRect.height-pad*2;

    if(Math.abs(dx)>0.01||Math.abs(dy)>0.01){
      // Project from center to edge
      const scaleX=dx!==0?(dx>0?w/2:-w/2)/dx:Infinity;
      const scaleY=dy!==0?(dy>0?h/2:-h/2)/dy:Infinity;
      const scale=Math.min(Math.abs(scaleX),Math.abs(scaleY));
      x=wrapperRect.width/2+dx*scale;
      y=wrapperRect.height/2+dy*scale;
    }else{
      x=wrapperRect.width/2;
      y=pad;
    }

    x=Math.max(pad,Math.min(wrapperRect.width-pad,x));
    y=Math.max(pad,Math.min(wrapperRect.height-pad,y));

    f.classList.remove('animating','fading');
    f.style.left=x+'px';
    f.style.top=y+'px';
    f.style.transform='translate(-50%,-50%)';
    f.classList.add('visible');
  });
};

// Combined horizontal + vertical drag scroll with PointerLock
document.querySelectorAll('.traceback-frames').forEach(c=>{
  if(c.dataset.dragInit)return;
  c.dataset.dragInit='1';
  let drag=false,scrollL,scrollT,vPanel=null;
  // Find the wrapper (parent)
  const wrapper=c.closest('.traceback-wrapper');
  // Initial update and scroll listener for floating symbols
  if(wrapper){
    updateFloatingSymbols(wrapper,c);
    c.addEventListener('scroll',()=>updateFloatingSymbols(wrapper,c));
    // Also update on vertical scroll within frame-content
    c.querySelectorAll('.frame-content').forEach(fc=>{
      fc.addEventListener('scroll',()=>updateFloatingSymbols(wrapper,c));
    });
    // Click handlers for floating symbols
    wrapper.querySelectorAll('.floating-symbol').forEach(f=>{
      f.addEventListener('click',()=>scrollto(f.dataset.frame));
    });
  }
  c.addEventListener('mousedown',e=>{
    if(e.button===0&&!e.target.closest('a,button,.floating-symbol')&&!e.ctrlKey&&!e.shiftKey&&!e.altKey&&!e.metaKey){
      drag=true;scrollL=c.scrollLeft;
      // Check if click is on a frame-content for vertical scrolling
      vPanel=e.target.closest('.frame-content');
      if(vPanel)scrollT=vPanel.scrollTop;
      c.style.cursor='grabbing';c.style.userSelect='none';
      c.style.scrollSnapType='none';
      c.style.scrollBehavior='auto';
      c.requestPointerLock();
      e.preventDefault();
    }
  });
  const endDrag=()=>{if(drag){drag=false;vPanel=null;c.style.cursor='';c.style.userSelect='';c.style.scrollBehavior='smooth';setTimeout(()=>c.style.scrollSnapType='',300);document.exitPointerLock()}};
  addEventListener('mouseup',endDrag);
  addEventListener('mousemove',e=>{
    if(drag){
      // Scroll only in dominant direction per frame
      if(Math.abs(e.movementX)>=Math.abs(e.movementY)||!vPanel){
        scrollL-=e.movementX;c.scrollLeft=scrollL;
      }else{
        scrollT-=e.movementY;vPanel.scrollTop=scrollT;
      }
    }
  });
  // Initially scroll each frame-content to show all marks (centered if they fit, else from top)
  c.querySelectorAll('.frame-content').forEach(d=>{
    const marks=d.querySelectorAll('mark,.tracerite-tooltip,.tracerite-symbol');
    if(marks.length){
      let minTop=Infinity,maxBottom=0;
      marks.forEach(m=>{
        const top=m.offsetTop;
        const bottom=top+m.offsetHeight;
        if(top<minTop)minTop=top;
        if(bottom>maxBottom)maxBottom=bottom;
      });
      const marksHeight=maxBottom-minTop;
      const panelHeight=d.clientHeight;
      // If marks fit, center them; otherwise scroll to show from first mark
      if(marksHeight<=panelHeight){
        const marksCenter=(minTop+maxBottom)/2;
        d.scrollTop=marksCenter-panelHeight/2;
      }else{
        d.scrollTop=minTop-panelHeight*0.1;  // Small top padding
      }
    }
  });
});
})()
