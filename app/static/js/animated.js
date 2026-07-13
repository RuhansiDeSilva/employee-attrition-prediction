// Animated landing page interactions
document.addEventListener('DOMContentLoaded', function(){
  const btn = document.getElementById('open-form');
  if(!btn) return;

  btn.addEventListener('click', function(e){
    // play a quick scale animation then navigate to the main form
    btn.style.transition = 'transform 220ms cubic-bezier(.2,.9,.3,1), opacity 220ms';
    btn.style.transform = 'scale(0.96)';
    btn.style.opacity = '0.95';

    setTimeout(()=>{
      // full reveal animation before redirect to the form
      window.location.href = '/form';
    }, 220);
  });
});
