from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)


TIER_COLORS = {
    "high": colors.HexColor("#e74c3c"),
    "medium": colors.HexColor("#f39c12"),
    "low": colors.HexColor("#27ae60"),
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallItalic", parent=styles["Italic"], fontSize=9, textColor=colors.grey))
    styles.add(ParagraphStyle(name="Banner", parent=styles["Heading2"], textColor=colors.white))
    styles.add(ParagraphStyle(name="RecTitle", parent=styles["Heading4"], spaceAfter=2))
    return styles


def _banner(tier, prob, summary, styles):
    color = TIER_COLORS.get(tier, colors.grey)
    text = f"<b>{tier.upper()} RISK</b> &nbsp;&nbsp; Churn probability: {prob*100:.1f}%"
    para = Paragraph(text, styles["Banner"])
    summary_para = Paragraph(summary, styles["BodyText"])
    table = Table([[para], [summary_para]], colWidths=[16 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), color),
        ("BACKGROUND", (0, 1), (0, 1), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _customer_table(customer):
    rows = [
        ["Total Spend", f"${customer['total_spend']:.0f}"],
        ["Support Calls", str(customer["support_calls"])],
        ["Payment Delay", f"{customer['payment_delay']} days"],
        ["Contract Length", customer["contract_length"]],
    ]
    table = Table(rows, colWidths=[5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.3, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _factor_items(factors, styles):
    items = []
    for f in factors:
        severity = f.get("severity", "medium").upper()
        name = f.get("name", "")
        evidence = f.get("evidence", "")
        text = f"<b>{name}</b> ({severity}) &mdash; {evidence}"
        items.append(ListItem(Paragraph(text, styles["BodyText"])))
    return ListFlowable(items, bulletType="bullet", leftIndent=12)


def build_pdf(report, customer):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Customer Retention Report",
    )
    styles = _styles()
    story = []

    story.append(Paragraph("Customer Retention Report", styles["Title"]))
    story.append(Paragraph(f"Generated on {date.today().isoformat()}", styles["SmallItalic"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Customer Snapshot", styles["Heading2"]))
    story.append(_customer_table(customer))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Risk Summary", styles["Heading2"]))
    story.append(_banner(
        report.get("risk_tier", "medium"),
        report.get("churn_probability", 0.0),
        report.get("risk_summary", ""),
        styles,
    ))
    story.append(Spacer(1, 0.4 * cm))

    factors = report.get("factors") or []
    if factors:
        story.append(Paragraph("Key Factors", styles["Heading2"]))
        story.append(_factor_items(factors, styles))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Recommended Actions", styles["Heading2"]))
    recs = report.get("recommendations") or []
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. {rec.get('action', '')}", styles["Heading4"]))
        story.append(Paragraph(f"<i>Why:</i> {rec.get('rationale', '')}", styles["BodyText"]))
        story.append(Paragraph(f"<i>Expected impact:</i> {rec.get('expected_impact', '')}", styles["BodyText"]))
        story.append(Paragraph(f"<i>Timeframe:</i> {rec.get('timeframe', '')}", styles["BodyText"]))
        story.append(Spacer(1, 0.25 * cm))

    sources = report.get("sources") or []
    if sources:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Sources", styles["Heading2"]))
        for s in sources:
            story.append(Paragraph(f"- {s['source']} &mdash; {s['section']}", styles["BodyText"]))

    note = report.get("confidence_note")
    if note:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"<i>Note:</i> {note}", styles["BodyText"]))

    disclaimer = report.get("disclaimer")
    if disclaimer:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(disclaimer, styles["SmallItalic"]))

    doc.build(story)
    return buf.getvalue()
