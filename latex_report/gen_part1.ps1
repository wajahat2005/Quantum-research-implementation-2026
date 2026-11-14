. "$PSScriptRoot\gen_common.ps1"
# ---------- macros ----------
$M=@(
 @('PimaRBFba','c','pima','Daga: RBF-SVM',0),@('PimaStackba','c','pima','Daga: OOF-Stacking (LR meta)',0),
 @('PimaAngleba','c','pima','Daga: Angle-QSVM',0),@('PimaZZba','c','pima','Daga: ZZ-QSVM',0),
 @('PimaAmpba','c','pima','Daga: Amplitude-QSVM',0),@('PimaKothaThrba','c','pima','Kotha: DiaBeats RF (threshold by inner CV)',0),
 @('PimaVQCba','c','pima','Kiran: HQCAN VQC (4q, 2 layers)',0),@('PimaHVba','c','pima','Daga: Hard-Voting (4 kernels)',0),
 @('PimaFloor','c','pima','Daga: RBF-SVM',5),
 @('SylRFba','c','sylhet','Daga: Random Forest',0),@('SylRBFba','c','sylhet','Daga: RBF-SVM',0),
 @('SylStackba','c','sylhet','Daga: OOF-Stacking (LR meta)',0),@('SylAngleba','c','sylhet','Daga: Angle-QSVM',0),
 @('SylZZba','c','sylhet','Daga: ZZ-QSVM',0),@('SylZZrec','c','sylhet','Daga: ZZ-QSVM',1),@('SylZZspec','c','sylhet','Daga: ZZ-QSVM',2),
 @('SylVQCba','c','sylhet','Kiran: HQCAN VQC (4q, 2 layers)',0),@('SylZZtunedba','c','sylhet','Yadav: ZZ-QSVM (linear) bw=inner-CV',0),
 @('SylFloor','c','sylhet','Daga: RBF-SVM',5),
 @('CdcKiranRBFba','c','cdc','Kiran: RBF-SVM (same 4 PCs, balanced)',0),@('CdcKiranRBFrec','c','cdc','Kiran: RBF-SVM (same 4 PCs, balanced)',1),
 @('CdcRBFrec','c','cdc','Daga: RBF-SVM',1),@('CdcRBFba','c','cdc','Daga: RBF-SVM',0),@('CdcFloor','c','cdc','Daga: RBF-SVM',5),
 @('HospBestba','c','hosp130','Kiran: Logistic Regression (same 4 PCs, balanced)',0),@('HospFloor','c','hosp130','Daga: RBF-SVM',5),
 @('UpamaSylRFclean','c','sylhet','Upama: RF (DML top-10)',4),@('UpamaSylRFpaper','p','sylhet','Upama: RF (DML top-10)',1),
 @('KothaPimaCleanAcc','c','pima','Kotha: DiaBeats RF',4),@('KothaPimaAUC','c','pima','Kotha: DiaBeats RF',3),
 @('KiranVQCcleanAcc','c','pima','Kiran: HQCAN VQC (4q, 2 layers)',4),@('KiranVQCpaperAcc','p','pima','Kiran: HQCAN VQC (4q, 2 layers)',1),
 @('YoungGMMba','c','pima','Young: QISICGM ensemble (+GMM, train-fold only)',0),@('YoungGMMrec','c','pima','Young: QISICGM ensemble (+GMM, train-fold only)',1),
 @('YoungGMMspec','c','pima','Young: QISICGM ensemble (+GMM, train-fold only)',2),@('YoungGMMauc','c','pima','Young: QISICGM ensemble (+GMM, train-fold only)',3),
 @('YoungGMMacc','c','pima','Young: QISICGM ensemble (+GMM, train-fold only)',4),
 @('YadavZpimaAcc','c','pima','Yadav: Z-QSVM bw=1',4),@('YadavZZpimaAcc','c','pima','Yadav: ZZ-QSVM (linear) bw=1',4),
 @('SodarAmpMinMaxRec','c','pima','Sodar: Amplitude-QSVM (MinMax, paper)',1),@('SodarAmpStdRec','c','pima','Sodar: Amplitude-QSVM (standardised)',1),
 @('DagaStackGap','p','pima','Daga: OOF-Stacking (LR meta)',2),
 @('DagaStackRec','c','pima','Daga: OOF-Stacking (LR meta)',1),@('DagaStackAUC','c','pima','Daga: OOF-Stacking (LR meta)',3))
$s=@()
foreach($e in $M){ $v= if($e[1] -eq 'c'){C $e[2] $e[3]}else{P $e[2] $e[3]}; $s+="\newcommand{\m$($e[0])}{$(F3 $v[$e[4]])}" }
foreach($ds in 'pima','sylhet','cdc','hosp130'){ foreach($vd in 'worse','better','n.s.'){ $n=@($tests|Where-Object{$_.ds -eq $ds -and $_.verdict -eq $vd}).Count; $k=($ds -replace '130','')+($vd -replace '\.','' -replace 'ns','NS'); $s+="\newcommand{\n$(($k -replace '[^A-Za-z]',''))}{$n}" } }
foreach($ds in 'pima','sylhet'){ foreach($m in 'LightGBM','LightGBM + GIF weights'){ $r=$zh|Where-Object{$_.ds -eq $ds -and $_.model -eq $m -and [double]$_.noise -eq 0.4}; $nm=($ds)+($m -replace '[^A-Za-z]',''); $s+="\newcommand{\zh$nm}{$(F3 $r.gmean)}" } }
$s | Set-Content "$out\macros.tex" -Encoding UTF8
# ---------- clean table ----------
$t=@()
foreach($ds in $dsName.Keys){
 $fl=(C $ds 'Daga: RBF-SVM')[5]
 $t+="\midrule\multicolumn{6}{l}{\textbf{$($dsName[$ds])} --- ``always guess the biggest group'' scores $(F3 $fl) accuracy}\\"
 $best=($sel|Where-Object{$_[0] -notmatch 'Majority'}|ForEach-Object{[double](C $ds $_[0])[0]}|Measure-Object -Maximum).Maximum
 foreach($e in $sel){ $v=C $ds $e[0]; $ba=F3 $v[0]; if([double]$v[0] -eq $best){$ba="\textbf{$ba}"}; $t+="$(X $e[0]) & $ba & $(F3 $v[1]) & $(F3 $v[2]) & $(F3 $v[3]) & $(F3 $v[4])\\" }
}
$t | Set-Content "$out\tab_clean.tex" -Encoding UTF8
# ---------- BA chart ----------
$c=@(); $ylist=($sel|ForEach-Object{$_[1]}) -join ','
$c+="\begin{tikzpicture}\begin{groupplot}[group style={group size=2 by 2,horizontal sep=0.5cm,vertical sep=1.7cm,yticklabels at=edge left},xbar,width=0.39\linewidth,height=6.6cm,xmin=0.4,xmax=1.0,/pgf/bar width=4pt,symbolic y coords={$ylist},ytick=data,y dir=reverse,yticklabel style={font=\tiny},xticklabel style={font=\tiny},title style={font=\small\bfseries},nodes near coords,nodes near coords style={font=\tiny,/pgf/number format/.cd,fixed,precision=2},point meta=x,extra x ticks={0.5},extra x tick labels={},extra x tick style={grid=major,grid style={dashed,gray}},clip=true]"
foreach($ds in $dsName.Keys){
 $c+="\nextgroupplot[title={$($dsName[$ds])}]"
 foreach($k in 'q','c'){ $co=($sel|Where-Object{$_[2] -eq $k}|ForEach-Object{"($(F3 (C $ds $_[0])[0]),$($_[1]))"}) -join ' '; $col= if($k -eq 'q'){'cq'}else{'cc'}; $c+="\addplot[fill=$col,draw=none] coordinates {$co};" }
}
$c+="\end{groupplot}\end{tikzpicture}"
$c | Set-Content "$out\fig_ba.tex" -Encoding UTF8
"part1 ok"; (Get-Content "$out\macros.tex").Count

