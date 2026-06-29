let datasets=[];let jobs=[];let selectedJob=null;
const $=id=>document.getElementById(id);
document.querySelectorAll('aside button').forEach(btn=>btn.onclick=()=>activateTab(btn.dataset.tab));
async function api(path,opts){const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.text()}
async function refresh(){
  datasets=await api('/api/datasets');jobs=await api('/api/jobs');
  $('datasetsList').innerHTML=datasets.map(d=>`<div class=card><b>${d.name}</b><p class=muted>${d.path}</p><p>RGB ${d.scan.image_count} | Depth ${d.scan.depth_count} | Video ${d.scan.video_count}</p><p>真实pose: ${d.scan.has_pose?'存在':'不存在'} | pose来源: ${d.scan.pose_source}</p><p class="risk risk-${d.scan.photo_risk}">照片级风险: ${d.scan.photo_risk}</p><div class=actions><button onclick="exportExrRgbd('${d.id}')">导出 EXR_RGBD</button></div></div>`).join('');
  $('datasetSelect').innerHTML=datasets.map(d=>`<option value="${d.id}">${d.name}</option>`).join('');
  $('jobsList').innerHTML=jobs.map(jobCard).join('');
}
function activateTab(tabId){
  document.querySelectorAll('aside button,.tab').forEach(x=>x.classList.remove('active'));
  document.querySelector(`aside button[data-tab="${tabId}"]`)?.classList.add('active');
  $(tabId).classList.add('active');
}
async function openJob(id){
  selectedJob=jobs.find(j=>j.id===id)||await api(`/api/jobs/${id}`);
  const artifacts=await api(`/api/jobs/${id}/artifacts`);
  const viewer=artifacts.find(a=>a.url&&a.url.endsWith('/viewer/index.html'));
  const qa=artifacts.find(a=>a.url&&a.url.endsWith('/source_view_qa.html'));
  if(viewer)$('previewFrame').src=viewer.url;
  if(qa)$('qaFrame').src=qa.url;
  activateTab('preview');
}
async function openQa(id){
  selectedJob=jobs.find(j=>j.id===id)||await api(`/api/jobs/${id}`);
  const artifacts=await api(`/api/jobs/${id}/artifacts`);
  const qa=artifacts.find(a=>a.url&&a.url.endsWith('/source_view_qa.html'));
  if(qa)$('qaFrame').src=qa.url;
  activateTab('qa');
}
async function showLogs(id){$('jobLogs').textContent=await api(`/api/jobs/${id}/logs`)}
async function cancelJob(id){await api(`/api/jobs/${id}/cancel`,{method:'POST'});await refresh();await showLogs(id)}
async function retryJob(id){const j=await api(`/api/jobs/${id}/retry`,{method:'POST'});await refresh();await showLogs(j.id)}
function downloadJob(id){window.open(`/api/jobs/${id}/download`,'_blank')}
function exportExrRgbd(id){window.open(`/api/datasets/${id}/export-exr-rgbd`,'_blank')}
function jobCard(j){
  const canOpen=j.status==='succeeded';
  const canCancel=j.status==='queued'||j.status==='running';
  const params=Object.keys(j.parameters||{}).length?`<details><summary>参数</summary><pre>${escapeHtml(JSON.stringify(j.parameters,null,2))}</pre></details>`:'';
  return `<div class="card job-card"><b>${templateLabel(j.template)}</b><span class="badge status-${j.status}">${j.status}</span><p class=muted>${j.id}</p><p class=muted>cancel: ${j.cancel_requested?'已请求':'否'}</p>${params}<div class=actions><button ${canOpen?'':'disabled'} onclick="openJob('${j.id}')">打开</button><button ${canOpen?'':'disabled'} onclick="openQa('${j.id}')">质检</button><button onclick="showLogs('${j.id}')">日志</button><button ${canCancel?'':'disabled'} onclick="cancelJob('${j.id}')">取消</button><button onclick="retryJob('${j.id}')">重试</button><button ${canOpen?'':'disabled'} onclick="downloadJob('${j.id}')">下载</button></div></div>`;
}
function templateLabel(value){return {quick_preview:'快速预览',standard:'标准重建',rgbd_optimized:'RGBD优化',high_quality:'高质量训练'}[value]||value}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function numberValue(id){const value=$(id).value;return value===''?undefined:Number(value)}
function textValue(id){const value=$(id).value.trim();return value||undefined}
function collectParams(){
  const params={};
  for(const [key,value] of Object.entries({
    scene_name:textValue('sceneName'),
    prompt:textValue('prompt'),
    colmap_max_image_size:numberValue('colmapSize'),
    artifixer_anchor_count:numberValue('artifixerAnchors'),
    artifixer_reconstruction_steps:numberValue('artifixerSteps'),
    max_point_count:numberValue('maxPointCount'),
    gsplat_max_steps:numberValue('gsplatSteps'),
    gsplat_max_frames:numberValue('gsplatFrames'),
    gsplat_image_max_size:numberValue('gsplatImageSize'),
    gsplat_max_points:numberValue('gsplatMaxPoints'),
    gsplat_target_gaussians:numberValue('gsplatTargetGaussians'),
    trained_ply:textValue('trainedPly'),
    training_metrics:textValue('trainingMetrics'),
  })){if(value!==undefined&&!Number.isNaN(value))params[key]=value}
  params.force_artifixer=$('forceArtifixer').checked;
  params.skip_artifixer=$('skipArtifixer').checked;
  params.train_gsplat=$('trainGsplat').checked;
  return params;
}
$('importPath').onclick=async()=>{await api('/api/datasets/import-path',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({path:$('datasetPath').value})});await refresh()};
$('uploadButton').onclick=async()=>{const f=$('uploadFile').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);await api('/api/datasets/upload',{method:'POST',body:fd});await refresh()};
$('createJob').onclick=async()=>{await api('/api/jobs',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({dataset_id:$('datasetSelect').value,template:$('templateSelect').value,parameters:collectParams()})});await refresh()};
$('doctorButton').onclick=async()=>$('doctorOutput').textContent=JSON.stringify(await api('/api/doctor'),null,2);
setInterval(refresh,4000);refresh().catch(e=>alert(e.message));
