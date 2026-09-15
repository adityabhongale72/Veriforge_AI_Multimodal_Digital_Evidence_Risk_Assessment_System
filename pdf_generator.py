import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_evidence_pdf(report_data: dict, output_pdf_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        fontName="Helvetica-Bold",
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a")
    )
    mono_style = ParagraphStyle(
        'MonospaceHash',
        parent=body_style,
        fontName="Courier",
        fontSize=7,
        leading=9
    )

    elements = []
    elements.append(Paragraph("DIGITAL FORENSIC EVIDENCE AUDIT REPORT", title_style))

    # Case Info
    elements.append(Paragraph("1. Chain of Custody & Evidence Metadata", heading_style))
    case_info = [
        [Paragraph("Case ID", body_bold), Paragraph(str(report_data.get("case_id", "NCFU-GEN")), body_style)],
        [Paragraph("File Name", body_bold), Paragraph(str(report_data.get("filename", "Evidence")), body_style)],
        [Paragraph("Media / File Type", body_bold), Paragraph(str(report_data.get("file_type", "Digital Media")), body_style)],
        [Paragraph("SHA-256 Hash", body_bold), Paragraph(str(report_data.get("sha256", "N/A")), mono_style)],
        [Paragraph("Acquisition Timestamp", body_bold), Paragraph(str(report_data.get("timestamp", "N/A")), body_style)]
    ]
    t1 = Table(case_info, colWidths=[150, 390])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t1)

    # Inferences
    elements.append(Paragraph("2. Forensic Model Inferences", heading_style))
    branch_a = report_data.get("branch_a", {})
    branch_b = report_data.get("branch_b", {})
    
    model_data = [
        [Paragraph("<b>Branch</b>", body_bold), Paragraph("<b>Target</b>", body_bold), Paragraph("<b>Score</b>", body_bold), Paragraph("<b>Findings</b>", body_bold)],
        [
            Paragraph("Branch A", body_style),
            Paragraph("Synthetic Generative Check", body_style),
            Paragraph(f"<b>{report_data.get('ai_probability', 'N/A')}</b>", body_style),
            Paragraph(str(branch_a.get("interpretation", "No generative anomalies detected.")), body_style)
        ],
        [
            Paragraph("Branch B", body_style),
            Paragraph("Tampering / Splicing Check", body_style),
            Paragraph(f"<b>{report_data.get('manipulation_probability', 'N/A')}</b>", body_style),
            Paragraph(str(branch_b.get("interpretation", "Coherent compression profile.")), body_style)
        ]
    ]
    t2 = Table(model_data, colWidths=[90, 140, 70, 240])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t2)

    # Visual Exhibit (ELA Heatmap if present)
    ela_mask_path = report_data.get("ela_mask_path", "")
    if ela_mask_path and os.path.exists(ela_mask_path):
        elements.append(Paragraph("3. Visual Exhibit: Error Level Analysis (ELA)", heading_style))
        ela_img = RLImage(ela_mask_path, width=280, height=160)
        ela_img.hAlign = 'CENTER'
        elements.append(KeepTogether([ela_img, Spacer(1, 4)]))

    # Verdict
    assessment = report_data.get("assessment", {})
    elements.append(Paragraph("4. Assessment & Final Verdict", heading_style))
    verdict_data = [
        [Paragraph("Authenticity Index", body_bold), Paragraph(f"<b>{report_data.get('authenticity', 'N/A')}</b>", body_style)],
        [Paragraph("Risk Classification", body_bold), Paragraph(f"<b>{assessment.get('risk_level', 'LOW')} RISK</b>", body_style)],
        [Paragraph("Final Classification", body_bold), Paragraph(f"<b>{report_data.get('verdict', 'N/A')}</b>", body_style)],
        [Paragraph("Summary", body_bold), Paragraph(str(report_data.get("legal_explanation", "Forensic audits complete.")), body_style)]
    ]
    t3 = Table(verdict_data, colWidths=[150, 390])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t3)

    doc.build(elements)
    return output_pdf_path