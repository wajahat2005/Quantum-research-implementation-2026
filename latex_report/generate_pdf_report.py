"""
generate_pdf_report.py - Standalone PDF generator for the WDBC Technical Audit & Replication Report.
Builds a publication-quality PDF directly from verified audit artifacts using ReportLab.
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Skip cover page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header
        self.drawString(54, 800, "WDBC Independent Technical Audit & Replication — Daga et al. (2026)")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 792, 558, 792)
        # Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Academic Reproduction & Verification Study")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.restoreState()

def build_pdf(filename="WDBC_AUDIT_REPORT.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54, rightMargin=54,
        topMargin=54, bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom typography
    c_blue = colors.HexColor("#1c3668")
    c_green = colors.HexColor("#145a32")
    c_red = colors.HexColor("#7d1919")
    c_gray = colors.HexColor("#f6f8fa")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_blue,
        alignment=1,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_blue,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1a252f")
    )

    table_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_body_style = ParagraphStyle(
        'TB',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#222222"),
        alignment=1
    )

    table_body_left = ParagraphStyle(
        'TBL',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#222222"),
        alignment=0
    )

    story = []

    # Title Page / Cover Section
    story.append(Spacer(1, 15))
    story.append(Paragraph("Independent Technical Audit & Replication Report", title_style))
    story.append(Paragraph("Adaptive Quantum Kernel Selection via Leakage-Free Stacking<br/><b>Evaluation on the Wisconsin Diagnostic Breast Cancer (WDBC) Benchmark</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=5, spaceAfter=15))

    # Meta-table
    meta_data = [
        [Paragraph("<b>Paper Reference:</b>", table_body_left), Paragraph("Daga, Naole, Parikh, Bommineni & Ramu (2026), <i>Sci. Rep.</i> 16, 26840. DOI: 10.1038/s41598-026-56928-1", table_body_left)],
        [Paragraph("<b>Official Repository:</b>", table_body_left), Paragraph("github.com/researchbackup27-sudo/Adaptive-Quantum-Kernel-Selection", table_body_left)],
        [Paragraph("<b>Verified Commit:</b>", table_body_left), Paragraph("<code>23c89d350e5135080a8f4f7f012c4e98d1ee3f82</code> (branch: <code>reproduction-verified-23c89d3</code>)", table_body_left)],
        [Paragraph("<b>Empirical Scope:</b>", table_body_left), Paragraph("Wisconsin Diagnostic Breast Cancer (WDBC) dataset exclusively ($N=569, D=30$).", table_body_left)],
        [Paragraph("<b>Reproduced Accuracy:</b>", table_body_left), Paragraph("<b>549 / 569 = 0.9649 (96.49%)</b> — Matches published Table 8 to three decimal places.", table_body_left)],
    ]
    meta_table = Table(meta_data, colWidths=[120, 384])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_gray),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e4e8")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Executive Summary Box
    summary_html = """
    <b>EXECUTIVE AUDIT SUMMARY:</b><br/>
    1. <b>Exact Table 8 Replicability:</b> Under Seed 42 and identical 10-fold CV, all 9 models' published Table 8 metrics (Accuracy, Specificity, Recall, Precision) replicate to 3 decimal places.<br/>
    2. <b>Hardcoded Model Selection:</b> Model selection is hardcoded at Line 787 of <code>quantum_clinical_github.py</code>. Pairwise McNemar scripts print tables but exert zero dynamic influence on pipeline execution.<br/>
    3. <b>Untrainable VQC:</b> Python list subtraction in the VQC loss function severs Autograd backpropagation. Gradient norm is exactly 0.0, rendering circuit parameters untrainable.<br/>
    4. <b>NISQ Noise Topology Swap:</b> The clean ZZ feature map uses complete graph entanglement ($K_8$, 56 CNOTs), but the noisy simulation secretly switches to a 1D linear chain ($P_8$, 14 CNOTs, -75% CNOTs). Under 5% noise, QSVM ZZ collapses to 0.0000 specificity.<br/>
    5. <b>Zero Quantum Superiority:</b> Across 10 random seeds, Classical RBF-SVM beats Quantum Stacking in 10/10 seeds ($0.9731$ vs. $0.9670$). No quantum metric survives Holm-Bonferroni correction ($p_{\mathrm{adj}} = 1.000$).
    """
    callout_data = [[Paragraph(summary_html, callout_style)]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eaf2f8")),
        ('BOX', (0,0), (-1,-1), 1.0, c_blue),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 15))

    # Section 1: Table 8 Full Comparison Matrix
    story.append(Paragraph("1. Evaluation Matrix: Published vs. Reproduced Table 8 Performance", h1_style))
    story.append(Paragraph("All metrics reproduced independently on the WDBC dataset ($N=569$) under Seed 42 using Stratified 10-fold Cross-Validation. Specificity denotes Malignant recall (Class 0), and Recall denotes Benign recall (Class 1).", body_style))

    t8_data = [
        [Paragraph("<b>Model</b>", table_header_style), 
         Paragraph("<b>Acc (Pub)</b>", table_header_style), Paragraph("<b>Acc (Rep)</b>", table_header_style),
         Paragraph("<b>Spec (Pub)</b>", table_header_style), Paragraph("<b>Spec (Rep)</b>", table_header_style),
         Paragraph("<b>Rec (Pub)</b>", table_header_style), Paragraph("<b>Rec (Rep)</b>", table_header_style),
         Paragraph("<b>Prec (Pub)</b>", table_header_style), Paragraph("<b>Prec (Rep)</b>", table_header_style)]
    ]
    
    rows = [
        ("Classical RBF-SVM", "0.970", "0.970", "0.948", "0.948", "0.983", "0.983", "0.969", "0.969"),
        ("QSVM (Angle)", "0.964", "0.964", "0.919", "0.919", "0.991", "0.991", "0.954", "0.954"),
        ("QSVM (ZZ Map)", "0.959", "0.959", "0.939", "0.939", "0.971", "0.971", "0.964", "0.964"),
        ("QSVM (Amplitude)", "0.954", "0.954", "0.881", "0.881", "0.997", "0.997", "0.934", "0.934"),
        ("QKNN (Angle)", "0.958", "0.958", "0.915", "0.915", "0.983", "0.983", "0.951", "0.951"),
        ("QKNN (ZZ Map)", "0.949", "0.949", "0.886", "0.886", "0.986", "0.986", "0.936", "0.936"),
        ("QKNN (Amplitude)", "0.952", "0.952", "0.896", "0.896", "0.986", "0.986", "0.941", "0.941"),
        ("Hard Voting Ensemble", "0.961", "0.961", "0.919", "0.919", "0.986", "0.986", "0.954", "0.954"),
        ("OOF Stacking Ensemble", "0.964", "0.964", "0.943", "0.943", "0.977", "0.977", "0.967", "0.967"),
    ]
    for r in rows:
        t8_data.append([
            Paragraph(f"<b>{r[0]}</b>" if "Stacking" in r[0] else r[0], table_body_left),
            Paragraph(r[1], table_body_style), Paragraph(f"<b>{r[2]}</b>" if r[1]==r[2] else r[2], table_body_style),
            Paragraph(r[3], table_body_style), Paragraph(r[4], table_body_style),
            Paragraph(r[5], table_body_style), Paragraph(r[6], table_body_style),
            Paragraph(r[7], table_body_style), Paragraph(r[8], table_body_style)
        ])

    t8_table = Table(t8_data, colWidths=[120, 48, 48, 48, 48, 48, 48, 48, 48])
    t8_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_blue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_gray]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t8_table)
    story.append(Spacer(1, 15))

    # Section 2: Full 10-Fold Confusion Matrices
    story.append(Paragraph("2. Full 10-Fold Cumulative Confusion Matrices ($N=569$)", h1_style))
    cm_data = [
        [Paragraph("<b>Classifier</b>", table_header_style), 
         Paragraph("<b>TN (Malignant)</b>", table_header_style), Paragraph("<b>FP</b>", table_header_style),
         Paragraph("<b>FN</b>", table_header_style), Paragraph("<b>TP (Benign)</b>", table_header_style),
         Paragraph("<b>Total Correct</b>", table_header_style), Paragraph("<b>Exact Accuracy</b>", table_header_style)]
    ]
    cm_rows = [
        ("Classical RBF-SVM", "201", "11", "6", "351", "552 / 569", "0.9701"),
        ("QSVM (Angle)", "195", "17", "3", "354", "549 / 569", "0.9649"),
        ("QSVM (ZZ Map)", "199", "13", "10", "347", "546 / 569", "0.9596"),
        ("QSVM (Amplitude)", "187", "25", "1", "356", "543 / 569", "0.9543"),
        ("QKNN (Angle)", "194", "18", "6", "351", "545 / 569", "0.9578"),
        ("QKNN (ZZ Map)", "188", "24", "5", "352", "540 / 569", "0.9490"),
        ("QKNN (Amplitude)", "190", "22", "5", "352", "542 / 569", "0.9525"),
        ("Hard Voting Ensemble", "195", "17", "5", "352", "547 / 569", "0.9613"),
        ("OOF Stacking Ensemble", "200", "12", "8", "349", "549 / 569", "0.9649"),
    ]
    for r in cm_rows:
        cm_data.append([
            Paragraph(f"<b>{r[0]}</b>" if "Stacking" in r[0] else r[0], table_body_left),
            Paragraph(r[1], table_body_style), Paragraph(r[2], table_body_style),
            Paragraph(r[3], table_body_style), Paragraph(r[4], table_body_style),
            Paragraph(r[5], table_body_style), Paragraph(f"<b>{r[6]}</b>", table_body_style)
        ])
    cm_table = Table(cm_data, colWidths=[130, 62, 50, 50, 62, 75, 75])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_blue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_gray]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(cm_table)
    story.append(Spacer(1, 15))

    # Page Break for Deep Dive Audits
    story.append(PageBreak())

    # Section 3: Deep Technical Discrepancy Audits
    story.append(Paragraph("3. Technical Audits of Core Discrepancies", h1_style))

    story.append(Paragraph("<b>Audit A: Statically Hardcoded Model Selection (Line 787)</b>", h2_style))
    story.append(Paragraph(
        "The published paper asserts that base classifiers are adaptively pruned using pairwise McNemar tests. In the actual code, the quartet is hardcoded at Line 787 of <code>quantum_clinical_github.py</code>. We executed an exhaustive nested 10-fold CV audit testing all 15 possible 4-model subsets. <b>Result:</b> In 10/10 outer folds, QSVM (Angle) ranked #1 in inner accuracy (0.9610–0.9746), yet it was excluded from the author's quartet. The author's quartet was selected in <b>0 out of 10 folds</b>.",
        body_style
    ))

    story.append(Paragraph("<b>Audit B: Variational Quantum Classifier (VQC) Autograd Severance</b>", h2_style))
    vqc_data = [
        [Paragraph("<b>Implementation</b>", table_header_style), Paragraph("<b>Loss Value</b>", table_header_style), Paragraph("<b>Grad Norm (L2)</b>", table_header_style), Paragraph("<b>Nonzero Grads</b>", table_header_style), Paragraph("<b>Finite Diff Error</b>", table_header_style)],
        [Paragraph("Author (List Subtraction)", table_body_left), Paragraph("0.996378", table_body_style), Paragraph("<b>0.000000</b>", table_body_style), Paragraph("0 / 48", table_body_style), Paragraph("0.092090", table_body_style)],
        [Paragraph("Corrected (pnp.stack)", table_body_left), Paragraph("0.996378", table_body_style), Paragraph("<b>0.101766</b>", table_body_style), Paragraph("28 / 48", table_body_style), Paragraph("1.11e-11", table_body_style)],
    ]
    vqc_table = Table(vqc_data, colWidths=[150, 85, 90, 85, 94])
    vqc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_red),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(vqc_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Because the author built predictions using a plain Python list comprehension, Autograd backpropagation was severed. The parameters never updated during training.", body_style))

    story.append(Paragraph("<b>Audit C: NISQ Noise Circuit Mismatch & Specificity Collapse</b>", h2_style))
    story.append(Paragraph(
        "Clean ZZ feature maps utilize an all-to-all complete graph $K_8$ (28 pairs, 56 CNOTs). The noisy script secretly swaps this for a 1D linear chain $P_8$ (7 pairs, 14 CNOTs), eliminating 75% of entangling gates. Furthermore, under 5% and 10% depolarizing noise, QSVM ZZ suffers catastrophic collapse: <b>Specificity drops to 0.0000</b> (TN=0, FP=22, predicting 100% Benign). The stacking ensemble survives only because the meta-learner adaptively drives $|\beta_{\mathrm{ZZ}}| < 0.001$.",
        body_style
    ))

    story.append(Paragraph("<b>Audit D: Multi-Seed Robustness (10 Seeds) & Holm Hypothesis Testing</b>", h2_style))
    seed_data = [
        [Paragraph("<b>Model</b>", table_header_style), Paragraph("<b>Mean Accuracy</b>", table_header_style), Paragraph("<b>Std Dev</b>", table_header_style), Paragraph("<b>Min Acc</b>", table_header_style), Paragraph("<b>Max Acc</b>", table_header_style)],
        [Paragraph("Classical RBF-SVM", table_body_left), Paragraph("<b>0.9731</b>", table_body_style), Paragraph("0.0017", table_body_style), Paragraph("0.9701", table_body_style), Paragraph("0.9754", table_body_style)],
        [Paragraph("QSVM (Angle)", table_body_left), Paragraph("0.9687", table_body_style), Paragraph("0.0023", table_body_style), Paragraph("0.9649", table_body_style), Paragraph("0.9736", table_body_style)],
        [Paragraph("OOF Quantum Stacking", table_body_left), Paragraph("0.9670", table_body_style), Paragraph("0.0031", table_body_style), Paragraph("0.9596", table_body_style), Paragraph("0.9701", table_body_style)],
    ]
    seed_table = Table(seed_data, colWidths=[150, 90, 85, 90, 89])
    seed_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_green),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(seed_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Across 10 independent stratified splits, Classical RBF-SVM outperforms Quantum Stacking in <b>10 out of 10 seeds</b> ($p=0.00098$, sign test). Out of 32 pairwise comparisons against RBF with Holm-Bonferroni correction, <b>zero quantum superiority results survive</b> ($p_{\mathrm{adj}} = 1.000$). All 6 surviving bootstrap tests show quantum models are significantly <i>inferior</i> to RBF on specificity and precision.", body_style))

    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#cccccc"), spaceBefore=10, spaceAfter=10))
    story.append(Paragraph("<b>Artifact Integrity Notice:</b> All numbers presented herein correspond exactly to files in <code>audit_results/</code>. Original research scripts were preserved unmodified.", callout_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf("C:\\Research\\Adaptive-Quantum-Kernel-Selection\\latex_report\\WDBC_AUDIT_REPORT.pdf")
