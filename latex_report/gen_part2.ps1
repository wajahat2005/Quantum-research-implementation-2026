. "$PSScriptRoot\gen_common.ps1"
# ---------- leakage table + chart (Sylhet) ----------
$lk=@(
 @('Yadav: ZZ-QSVM (full) bw=1','ZZ QSVM full'),@('Yadav: ZZ-QSVM (linear) bw=1','ZZ QSVM linear'),@('Yadav: Z-QSVM bw=1','Z QSVM'),
 @('Daga: Amplitude-QSVM','Amplitude QSVM'),@('Zhang: LightGBM + GIF weights','LightGBM GIF'),@('Kotha: DiaBeats RF','Kotha RF'),
 @('Daga: Random Forest','Random Forest'),@('Daga: RBF-SVM','RBF-SVM'),@('Daga: OOF-Stacking (LR meta)','Stacking Daga'),@('Daga: Angle-QSVM','Angle QSVM'))
$t=@(); $cp=@(); $cc=@()
foreach($e in $lk){ $v=P 'sylhet' $e[0]; $g=([double]$v[2]*100).ToString('+0.0;-0.0',$inv)
 $t+="$(X $e[0]) & $(F3 $v[1]) & $(F3 $v[0]) & $g \\"; $cp+="($(F3 $v[1]),$($e[1]))"; $cc+="($(F3 $v[0]),$($e[1]))" }
$t | Set-Content "$out\tab_leak.tex" -Encoding UTF8
$yl=($lk|ForEach-Object{$_[1]}) -join ','
@("\begin{tikzpicture}\begin{axis}[xbar,width=0.8\linewidth,height=8cm,xmin=0.6,xmax=1.02,/pgf/bar width=5pt,symbolic y coords={$yl},ytick=data,y dir=reverse,yticklabel style={font=\scriptsize},xticklabel style={font=\scriptsize},xlabel={accuracy on Sylhet},legend style={at={(0.5,-0.18)},anchor=north,legend columns=2,font=\scriptsize},nodes near coords,nodes near coords style={font=\tiny,/pgf/number format/.cd,fixed,precision=2},point meta=x]",
 "\addplot[fill=cp,draw=none] coordinates {$($cp -join ' ')};","\addplot[fill=cc,draw=none] coordinates {$($cc -join ' ')};",
 "\legend{paper-style split (duplicates leak),clean grouped split}","\end{axis}\end{tikzpicture}") | Set-Content "$out\fig_leak.tex" -Encoding UTF8
# ---------- recall chart cdc / hosp ----------
$c1=@(); $c2=@(); $ylist=($sel|Where-Object{$_[0] -notmatch 'Majority'}|ForEach-Object{$_[1]}) -join ','
foreach($e in ($sel|Where-Object{$_[0] -notmatch 'Majority'})){ $c1+="($(F3 (C 'cdc' $e[0])[1]),$($e[1]))"; $c2+="($(F3 (C 'hosp130' $e[0])[1]),$($e[1]))" }
@("\begin{tikzpicture}\begin{axis}[xbar,width=0.8\linewidth,height=7.5cm,xmin=0,xmax=1,/pgf/bar width=4pt,symbolic y coords={$ylist},ytick=data,y dir=reverse,yticklabel style={font=\scriptsize},xticklabel style={font=\scriptsize},xlabel={recall (share of sick patients found)},legend style={at={(0.5,-0.2)},anchor=north,legend columns=2,font=\scriptsize}]",
 "\addplot[fill=cp,draw=none] coordinates {$($c1 -join ' ')};","\addplot[fill=cc,draw=none] coordinates {$($c2 -join ' ')};","\legend{CDC,Hospital 130}","\end{axis}\end{tikzpicture}") | Set-Content "$out\fig_recall.tex" -Encoding UTF8
# ---------- tests table ----------
$tm=@('Daga: OOF-Stacking (LR meta)','Daga: Angle-QSVM','Daga: ZZ-QSVM','Daga: Amplitude-QSVM','Daga: Hard-Voting (4 kernels)','Yadav: ZZ-QSVM (linear) bw=inner-CV','Yadav: ZZ-QSVM (linear) bw=1','Kiran: HQCAN VQC (4q, 2 layers)','Kiran: RBF-SVM (same 4 PCs, balanced)','Kotha: DiaBeats RF (threshold by inner CV)')
$t=@()
foreach($ds in $dsName.Keys){ $t+="\midrule\multicolumn{5}{l}{\textbf{$($dsName[$ds])}}\\"
 foreach($m in $tm){ $r=T $ds $m; if(-not $r){continue}
  $ph= if([double]$r.ph -lt 0.0001){'$<$0.0001'}else{F3 $r.ph}
  $vd= switch($r.verdict){'worse'{'\BAD{worse}'}'better'{'\OK{better}'}default{'no clear difference'}}
  $t+="$(X $m) & $(([double]$r.diff).ToString('+0.000;-0.000',$inv)) & $(F3 $r.p) & $ph & $vd\\" } }
$t | Set-Content "$out\tab_tests.tex" -Encoding UTF8
# ---------- zhang chart ----------
$ax=@()
foreach($ds in 'pima','sylhet'){ $ax+="\nextgroupplot[title={$($dsName[$ds])}]"
 foreach($m in 'LightGBM','LightGBM + GIF weights'){ $co=($zh|Where-Object{$_.ds -eq $ds -and $_.model -eq $m}|Sort-Object {[double]$_.noise}|ForEach-Object{"($(([double]$_.noise*100).ToString('0',$inv)),$(F3 $_.gmean))"}) -join ' '; $col= if($m -eq 'LightGBM'){'cc'}else{'cq'}; $ax+="\addplot[color=$col,mark=*,thick] coordinates {$co};" } }
@("\begin{tikzpicture}\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.6cm},width=0.46\linewidth,height=5.5cm,xlabel={share of wrong labels (\%)},ylabel={G-mean},ymin=0.5,ymax=0.95,grid=major,tick label style={font=\scriptsize},title style={font=\small\bfseries},legend style={font=\scriptsize,at={(1,1.35)},anchor=south east,legend columns=2}]")+$ax+@("\legend{plain LightGBM,LightGBM + GIF weights}","\end{groupplot}\end{tikzpicture}") | Set-Content "$out\fig_zhang.tex" -Encoding UTF8
# ---------- claims table ----------
$cl=@(
 @('Daga','Pima, stacking accuracy',0.766,'b','pima','Daga: OOF-Stacking (LR meta)'),
 @('Daga','Pima, Angle QSVM accuracy',0.763,'b','pima','Daga: Angle-QSVM'),
 @('Daga','Pima, ZZ QSVM accuracy',0.759,'b','pima','Daga: ZZ-QSVM'),
 @('Daga','Pima, Amplitude QSVM accuracy',0.763,'b','pima','Daga: Amplitude-QSVM'),
 @('Daga','Pima, RBF-SVM accuracy',0.758,'b','pima','Daga: RBF-SVM'),
 @('Upama','Pima, SVM accuracy',0.72,'b','pima','Upama: Linear SVM (DML top-7)'),
 @('Upama','Pima, QSVM (our Angle)',0.65,'b','pima','Upama: Angle-QSVM (DML top-7)'),
 @('Upama','Pima, QSVM (our ZZ, width 1)',0.65,'b','pima','Yadav: ZZ-QSVM (full) bw=1'),
 @('Upama','Pima, RF accuracy',0.76,'b','pima','Upama: RF (DML top-7)'),
 @('Upama','Sylhet, SVM accuracy',0.9231,'b','sylhet','Upama: Linear SVM (DML top-10)'),
 @('Upama','Sylhet, RF accuracy',0.9712,'b','sylhet','Upama: RF (DML top-10)'),
 @('Upama','Sylhet, QSVM (our Angle)',0.9712,'b','sylhet','Upama: Angle-QSVM (DML top-10)'),
 @('Upama','Sylhet, QSVM (our ZZ, width 1)',0.9712,'b','sylhet','Yadav: ZZ-QSVM (full) bw=1'),
 @('Kotha','Pima, RF accuracy',0.8333,'b','pima','Kotha: DiaBeats RF'),
 @('Kiran','Pima, VQC accuracy',0.997,'b','pima','Kiran: HQCAN VQC (4q, 2 layers)'))
$t=@()
foreach($e in $cl){ $r=B $e[4] $e[5]; $a=[double]$r.acc; $lo=[double]$r.lo; $hi=[double]$r.hi; $d=($a-$e[2]).ToString('+0.000;-0.000',$inv)
 $vd= if($e[2] -ge $lo -and $e[2] -le $hi){'\OK{claim inside our 95\% range}'}elseif($e[2] -gt $hi){'\BAD{claim too high to reproduce}'}else{'\WARN{we score higher than claim}'}
 $t+="$($e[0]) & $($e[1]) & $($r.n) & $((F3 $e[2])) & $(F3 $a) [$(F3 $lo), $(F3 $hi)] & $d & $vd\\" }
$t | Set-Content "$out\tab_claims.tex" -Encoding UTF8
Copy-Item "$env:TEMP\nb2_exec\results\09_heatmap_balanced_accuracy.png" $out -Force
"part2 ok"

Get-ChildItem $out -Filter *.tex | ForEach-Object { $t=[IO.File]::ReadAllText($_.FullName); [IO.File]::WriteAllText($_.FullName,$t.TrimStart([char]0xFEFF),(New-Object Text.UTF8Encoding $false)) }
foreach($n in "tab_clean","tab_tests"){ $p="$out\$n.tex"; $t=[IO.File]::ReadAllText($p); if($t.StartsWith("\midrule")){ [IO.File]::WriteAllText($p,$t.Substring(8),(New-Object Text.UTF8Encoding $false)) } }
$wr=@{ tab_clean=@('{lccccc}','Model & Bal.\ acc & Recall & Specificity & AUC & Accuracy\\\midrule'); tab_tests=@('{lcccl}','Model & Mean diff & Raw $p$ & Holm $p$ & Verdict\\\midrule'); tab_leak=@('{lccc}','Model & Paper-style & Clean & Gap (points)\\\midrule'); tab_claims=@('{llrcccl}','Paper & Setting & N & Claimed & Ours [95\%] & Diff & Verdict\\\midrule') }
foreach($n in $wr.Keys){ $t=[IO.File]::ReadAllText("$out\$n.tex"); $t=$t.TrimStart(); $full="\begin{tabular}$($wr[$n][0])`n\toprule $($wr[$n][1])`n$t`n\bottomrule\end{tabular}`n"; [IO.File]::WriteAllText("$out\w_$n.tex",$full,(New-Object Text.UTF8Encoding $false)) }
