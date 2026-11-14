$w="$env:TEMP\nb2_work"
$out='D:\research\New folder\Comparative approch of daga\Adaptive-Quantum-Kernel-Selection\latex_report\gen_v2'
New-Item -ItemType Directory -Force $out | Out-Null
$clean=Import-Csv "$w\clean.csv"; $pv=Import-Csv "$w\pv.csv"; $tests=Import-Csv "$w\tests.csv"; $boot=Import-Csv "$w\boot.csv"; $zh=Import-Csv "$w\zhang.csv"
$inv=[cultureinfo]::InvariantCulture
function F3($x){ ([double]$x).ToString('0.000',$inv) }
function F2($x){ ([double]$x).ToString('0.00',$inv) }
function X($s){ $s -replace '([&%_#])','\$1' }
function C($ds,$m){ $r=$clean|Where-Object{$_.ds -eq $ds -and $_.model -eq $m}|Select-Object -First 1; if(-not $r){throw "missing clean $ds $m"}; ,($r.v -split ';') }
function P($ds,$m){ $r=$pv|Where-Object{$_.ds -eq $ds -and $_.model -eq $m}|Select-Object -First 1; if(-not $r){throw "missing pv $ds $m"}; ,($r.v -split ';') }
function B($ds,$m){ $r=$boot|Where-Object{$_.ds -eq $ds -and $_.model -eq $m}|Select-Object -First 1; if(-not $r){throw "missing boot $ds $m"}; $r }
function T($ds,$m){ $tests|Where-Object{$_.ds -eq $ds -and $_.model -eq $m}|Select-Object -First 1 }
# selected models: full name, short chart label, q/c
$sel=@(
 @('Kotha: DiaBeats RF (threshold by inner CV)','RF tuned threshold','c'),
 @('Daga: Random Forest','Random Forest','c'),
 @('Daga: Logistic Regression','Logistic Regression','c'),
 @('Daga: RBF-SVM','RBF-SVM ref','c'),
 @('Daga: OOF-Stacking (LR meta)','Stacking Daga','q'),
 @('Daga: Angle-QSVM','Angle QSVM','q'),
 @('Yadav: ZZ-QSVM (linear) bw=inner-CV','ZZ QSVM tuned width','q'),
 @('Daga: ZZ-QSVM','ZZ QSVM default width','q'),
 @('Daga: Amplitude-QSVM','Amplitude QSVM','q'),
 @('Daga: Hard-Voting (4 kernels)','Hard voting 4 kernels','q'),
 @('Kiran: HQCAN VQC (4q, 2 layers)','VQC Kiran','q'),
 @('Daga: Majority class','Always majority','c'))
$dsName=[ordered]@{pima='Pima (768 rows)';sylhet='Sylhet (520 rows)';cdc='CDC (2,000 rows)';hosp130='Hospital 130 (2,000 rows)'}
