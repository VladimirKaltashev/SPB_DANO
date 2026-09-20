/** Build editable PowerPoint charts from monthly_offences.py's deck_data.json.
 * Usage: node build_monthly_deck.mjs INPUT_JSON OUTPUT_PPTX
 * Requires @oai/artifact-tool and PRESENTATIONS_SKILL_DIR for validation.
 * RUNTIME_PYTHON may select the bundled Python validation runtime.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const input=path.resolve(process.argv[2] ?? 'outputs/monthly_offences/deck_data.json');
const finalPath=path.resolve(process.argv[3] ?? 'outputs/monthly_offences/monthly_fines.pptx');
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if(!skill) throw new Error('Set PRESENTATIONS_SKILL_DIR to the installed presentations skill');
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const data=JSON.parse(await fs.readFile(input,'utf8'));
const workspaceDir=path.resolve(process.env.TASK_WORKSPACE??process.cwd());
const buildDir=path.join(workspaceDir,'tmp/monthly-pptx-build');
await fs.mkdir(buildDir,{recursive:true});
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const family=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const chartOwners=[];
function text(slide,value,x,y,w,h,size=24,color='#203444',bold=false){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=value;
  s.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};
  return s;
}
for(let i=0;i<data.chart_specs.length;i++){
  const spec=data.chart_specs[i];
  const slide=p.slides.add();
  slide.background.fill='#FFFFFF';
  text(slide,spec.title,48,24,1184,62,42,'#172D40',true);
  text(slide,spec.subtitle,50,92,1170,44,23,'#526776');
  const grouped=spec.grouping==='clustered';
  const totals=spec.months.map((_,j)=>spec.series.reduce((n,s)=>n+s.values[j],0));
  const maximum=grouped?Math.max(...spec.series.flatMap(s=>s.values)):Math.max(...totals);
  const multiple=spec.series.length>1;
  const chart=slide.charts.add('bar',{
    position:{left:58,top:163,width:1170,height:multiple?468:467},
    categories:spec.month_labels,
    // Excel stores 15 significant digits. Six decimals are well beyond the
    // displayed 0.1 precision and avoid lossy workbook serialization failures.
    series:spec.series.map(s=>({name:s.name,values:s.values.map(v=>Number(v.toFixed(6))),fill:s.color,line:{fill:'none',width:0},valuesFormatCode:spec.kind==='category_focus'?'0.0':'#,##0'})),
    barOptions:{direction:'column',grouping:grouped?'clustered':multiple?'stacked':'clustered',gapWidth:70,overlap:grouped?0:100},
    hasLegend:multiple,
    legend:{position:'bottom',overlay:false,textStyle:{typeface:family,fontSize:20,fill:'#334B5C'}},
    xAxis:{visible:true,numberFormatCode:'@',textStyle:{typeface:family,fontSize:22,fill:'#334B5C'},majorGridlines:null,line:{fill:'#A9B8C2',width:1}},
    yAxis:{visible:true,min:0,max:Math.max(maximum*1.22,1),numberFormatCode:spec.kind==='category_focus'?'0.0':'#,##0',textStyle:{typeface:family,fontSize:19,fill:'#526776'},majorGridlines:{fill:'#E5EBEF',width:1},line:{fill:'none',width:0}},
    dataLabels:{showValue:!multiple,position:'outEnd',textStyle:{typeface:family,fontSize:23,fill:'#203444'}},
    chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},
  });
  applyPresentationChartFont(chart,{fontFamily:family});
  text(slide,spec.y_title??'Число штрафов, шт.',70,137,690,30,21,'#526776');
  // Annotation is an editable shape. It marks the boundary after two months.
  // The PNG export uses an exact data-axis position; this PPTX line is approximate.
  const crisisX=130+(1205-130)*(data.crisis_index+.5)/spec.months.length;
  slide.shapes.add({geometry:'line',position:{left:crisisX,top:200,width:0,height:multiple?(grouped?355:338):389},fill:'none',line:{fill:'#C04538',width:2,style:'dashed'}});
  text(slide,'≈ начало кризиса '+spec.crisis_label,crisisX+12,171,470,28,20,'#B23F35');
  text(slide,spec.note,50,650,1135,50,18,'#526776');
  text(slide,String(i+1),1190,667,55,32,16,'#7B8F9C');
  const note={source:'data/processed/fines_clean.csv; outputs/behavior_cluster_analysis/client_behavior_features.csv',period:[data.start,data.end_exclusive],crisis:data.crisis,cohort:spec.kind==='category_focus'?'fixed cohort, subscription <= start':'all assigned clients',count:'unique bill_id by offence date, not amount',categories:spec.series.map(s=>({label:s.name,source:s.full_name??s.name,values:s.values})),method:data.method};
  slide.speakerNotes.textFrame.setText(JSON.stringify(note,null,2));
  chartOwners.push(i+1);
}
const candidatePath=path.join(buildDir,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const result=await finalizePresentation({
  workspaceDir,candidatePath,finalPath,
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  requiredNativeChartOwnerSlides:chartOwners,
  materializeLiteralChartWorkbooks:true,
  nativeChartTargetApplication:'portable',
  fontPolicy:{basis:'design',families:[family]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,`${path.basename(finalPath)}.validation.json`),
});
console.log(JSON.stringify({finalPath:result.finalPath,slides:data.chart_specs.length,bytes:result.byteCount,receipt:result.receiptPath},null,2));
