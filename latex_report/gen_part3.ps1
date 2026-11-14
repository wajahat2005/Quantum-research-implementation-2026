# Master comparison table: paper method x dataset (clean protocol), numbers from %TEMP%\nb2_work\clean.csv
# Run inline (execution policy blocks scripts): & ([scriptblock]::Create((Get-Content gen_part3.ps1 -Raw)))
$w="$env:TEMP\nb2_work"; $out='D:\research\New folder\Comparative approch of daga\Adaptive-Quantum-Kernel-Selection\latex_report\gen_v2'
$clean=Import-Csv "$w\clean.csv"
$BS='\'; $DBS='\\'
function V($ds,$m){
  $r=$clean|?{$_.ds -eq $ds -and ($_.model -eq $m -or $_.model -eq $m.Replace('top-K','top-7') -or $_.model -eq $m.Replace('top-K','top-10'))}|select -First 1
  if(-not $r){throw "missing $ds $m"}; $p=$r.v -split ';'
  @{ba=[double]$p[0];rec=[double]$p[1];spec=[double]$p[2];auc=[double]$p[3];acc=[double]$p[4];fl=[double]$p[5]} }
$rows=@(
 @('Upama','Angle QSVM, DML top features','Upama: Angle-QSVM (DML top-K)'),
 @('Upama','Random forest, DML top features','Upama: RF (DML top-K)'),
 @('Young','QISICGM ensemble + GMM (train folds only)','Young: QISICGM ensemble (+GMM, train-fold only)'),
 @('Kotha','Random forest','Kotha: DiaBeats RF'),
 @('Kotha','Random forest, tuned threshold','Kotha: DiaBeats RF (threshold by inner CV)'),
 @('Zhang','LightGBM','Zhang: LightGBM'),
 @('Zhang','LightGBM + GIF weights','Zhang: LightGBM + GIF weights'),
 @('Kiran','VQC (4 qubits)','Kiran: HQCAN VQC (4q, 2 layers)'),
 @('Kiran','RBF-SVM, same 4 PCs, balanced','Kiran: RBF-SVM (same 4 PCs, balanced)'),
 @('Sodar','Angle QSVM','Sodar: Angle-QSVM'),
 @('Sodar','Amplitude QSVM (standardised)','Sodar: Amplitude-QSVM (standardised)'),
 @('Yadav','Z QSVM, width by inner CV','Yadav: Z-QSVM bw=inner-CV'),
 @('Yadav','ZZ QSVM (linear), width by inner CV','Yadav: ZZ-QSVM (linear) bw=inner-CV'),
 @('Daga','Angle QSVM','Daga: Angle-QSVM'),
 @('Daga','ZZ QSVM (default width)','Daga: ZZ-QSVM'),
 @('Daga','Stacking (OOF, LR meta)','Daga: OOF-Stacking (LR meta)'),
 @('Ref.','RBF-SVM (reference)','Daga: RBF-SVM'),
 @('Ref.','Always guess biggest group','Daga: Majority class'))
$inv=[cultureinfo]::InvariantCulture
$ML=$BS+'midrule'
$t=$BS+'begin{tabular}{llcccc}'+"`n"+$BS+'toprule & & Pima & Sylhet & CDC & Hosp.'+$BS+' 130'+$DBS+"`n"+'Paper & Method & '+$BS+'multicolumn{4}{c}{balanced acc.'+$BS+' / recall}'+$DBS+$ML+"`n"
$prev=''
foreach($r in $rows){
  if($prev -ne '' -and $prev -ne $r[0]){ $t+=$ML+"`n" }; $prev=$r[0]
  $c=@(); foreach($d in 'pima','sylhet','cdc','hosp130'){ $v=V $d $r[2]; $c+=('{0} / {1}' -f $v.ba.ToString('0.00',$inv),$v.rec.ToString('0.00',$inv)) }
  $t+=($r[0]+' & '+$r[1]+' & '+($c -join ' & ')+$DBS)+"`n" }
$fl=@(); foreach($d in 'pima','sylhet','cdc','hosp130'){ $fl+=(V $d 'Daga: Majority class').fl.ToString('0.00',$inv) }
$t+=$ML+"`n"+$BS+'multicolumn{2}{l}{Accuracy of always guessing the biggest group} & '+($fl -join ' & ')+$DBS+"`n"+$BS+'bottomrule'+$BS+'end{tabular}'
Set-Content "$out\tab_master.tex" $t -Encoding utf8
