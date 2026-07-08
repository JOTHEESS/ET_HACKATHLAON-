"""
Synthetic industrial document corpus generator.
Builds a pump-focused corpus with a PLANTED multi-hop near-miss chain
that only a knowledge graph can traverse (keyword search fails on it).

Seed fixed at 42 for reproducibility.
"""
import os
import random
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "data", "corpus", "synthetic")
os.makedirs(OUT, exist_ok=True)

styles = getSampleStyleSheet()
H = ParagraphStyle('H', parent=styles['Heading1'], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#1F3864"))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=6, textColor=colors.HexColor("#2F5496"))
BODY = ParagraphStyle('BODY', parent=styles['Normal'], fontSize=10, leading=15, spaceAfter=6)
META = ParagraphStyle('META', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor("#666666"))


def make_pdf(filename, flowables):
    path = os.path.join(OUT, filename)
    doc = SimpleDocTemplate(path, pagesize=letter,
                            topMargin=0.8*inch, bottomMargin=0.8*inch,
                            leftMargin=0.9*inch, rightMargin=0.9*inch)
    doc.build(flowables)
    return path


def kv_table(rows):
    t = Table(rows, colWidths=[1.8*inch, 4.3*inch])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#1F3864")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    return t


# ============================================================
# THE PLANTED NEAR-MISS CHAIN (star demo)
# Timeline: failure on 2024-07-22. Warnings start 3 weeks prior.
# ============================================================
FAIL_DATE = datetime(2024, 7, 22)
IR_DATE = FAIL_DATE - timedelta(days=21)   # inspection: early warning
ML_DATE = FAIL_DATE - timedelta(days=14)   # maint log: deferred fix
VS_DATE = FAIL_DATE - timedelta(days=5)    # sensor: threshold crossed

D = "%d %b %Y"

# -------- DOC 1: Inspection Report IR-556 (EARLY WARNING) --------
def doc_inspection_ir556():
    f = []
    f.append(Paragraph("INSPECTION REPORT", H))
    f.append(Paragraph("Document Ref: IR-556  |  Rotating Equipment Inspection", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Inspection Date", IR_DATE.strftime(D)],
        ["Asset", "Centrifugal Pump P-204 (Boiler Feed Water Service)"],
        ["Location", "Unit 2, Pump House Bay 4"],
        ["Inspector", "R. Krishnan (Cert. Vibration Analyst, ISO 18436-2 Cat II)"],
        ["Related Procedure", "M-118 (Rotating Equipment Condition Monitoring)"],
    ]))
    f.append(Spacer(1, 10))
    f.append(Paragraph("Findings", H2))
    f.append(Paragraph(
        "Routine condition-monitoring inspection of Pump P-204 completed. Drive-end (DE) bearing "
        "vibration measured at 4.6 mm/s RMS, up from 2.1 mm/s RMS at the previous quarter reading. "
        "This is within the ISO 10816-3 Zone C band (elevated) but below the Zone D alarm threshold "
        "of 7.1 mm/s. Bearing housing temperature nominal at 62 C. Lubrication level low.", BODY))
    f.append(Paragraph(
        "The rising vibration trend on the DE bearing is consistent with early-stage bearing wear. "
        "This is a leading indicator and warrants tracking. No immediate shutdown required.", BODY))
    f.append(Paragraph("Recommendation", H2))
    f.append(Paragraph(
        "1. Flag P-204 DE bearing for enhanced monitoring (weekly, not quarterly). "
        "2. Schedule bearing replacement at next maintenance window per Procedure M-118. "
        "3. Top up lubrication in the interim. "
        "Recommendation status: ELEVATED - MONITOR. Escalate if amplitude exceeds 7.1 mm/s.", BODY))
    make_pdf("IR-556_inspection_report.pdf", f)


# -------- DOC 2: Maintenance Log ML-1183 (DEFERRED FIX) --------
def doc_maintlog_ml1183():
    f = []
    f.append(Paragraph("MAINTENANCE WORK ORDER / LOG", H))
    f.append(Paragraph("Document Ref: ML-1183  |  Corrective Maintenance", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Work Date", ML_DATE.strftime(D)],
        ["Asset", "Pump P-204"],
        ["Triggered By", "Inspection Report IR-556"],
        ["Technician", "S. Prakash (Mech. Maint. Team B)"],
        ["Supervisor Sign-off", "V. Menon"],
        ["Procedure Referenced", "M-118"],
    ]))
    f.append(Spacer(1, 10))
    f.append(Paragraph("Work Performed", H2))
    f.append(Paragraph(
        "Attended P-204 following IR-556 recommendation. Lubrication topped up on DE bearing housing "
        "with grade ISO VG 68. Vibration re-checked at 4.9 mm/s RMS (marginally higher than IR-556 "
        "reading). Bearing replacement was NOT carried out during this visit.", BODY))
    f.append(Paragraph("Deferral Note", H2))
    f.append(Paragraph(
        "DE bearing replacement deferred to next planned shutdown due to spare bearing (SKF 6316 C3) "
        "not being in stores at time of work order. Lube top-up applied as interim measure. "
        "Deferral approved by V. Menon. Next planned shutdown scheduled 6 weeks out.", BODY))
    f.append(Paragraph(
        "NOTE: Procedure M-118 clause 4.3 states elevated-vibration bearings flagged in inspection "
        "should be replaced within 14 days, not deferred to next shutdown. Deferral logged as a "
        "temporary deviation pending spare availability.", BODY))
    make_pdf("ML-1183_maintenance_log.pdf", f)


# -------- DOC 3: Vibration Sensor Log VS-204 (THRESHOLD CROSSED) --------
def doc_sensorlog_vs204():
    f = []
    f.append(Paragraph("VIBRATION MONITORING TREND LOG", H))
    f.append(Paragraph("Document Ref: VS-204  |  Online Condition Monitoring", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Asset", "Pump P-204 - DE Bearing"],
        ["Sensor Tag", "VT-204-DE (accelerometer, mm/s RMS)"],
        ["Monitoring Mode", "Continuous online (post IR-556 escalation)"],
        ["Alarm Threshold", "7.1 mm/s RMS (ISO 10816-3 Zone D)"],
    ]))
    f.append(Spacer(1, 10))
    f.append(Paragraph("Trend Data (weekly RMS)", H2))
    trend = [
        ["Date", "DE Bearing (mm/s RMS)", "Status"],
        [IR_DATE.strftime(D), "4.6", "Zone C - Elevated"],
        [ML_DATE.strftime(D), "4.9", "Zone C - Elevated"],
        [(ML_DATE + timedelta(days=4)).strftime(D), "5.8", "Zone C - Rising"],
        [VS_DATE.strftime(D), "7.4", "Zone D - ALARM"],
        [(VS_DATE + timedelta(days=2)).strftime(D), "9.1", "Zone D - CRITICAL"],
    ]
    t = Table(trend, colWidths=[1.7*inch, 2.2*inch, 2.2*inch])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 4), (-1, 5), colors.HexColor("#F8D7DA")),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    f.append(t)
    f.append(Spacer(1, 10))
    f.append(Paragraph(
        "DE bearing vibration crossed the 7.1 mm/s Zone D alarm threshold on " + VS_DATE.strftime(D) +
        ", five days before failure. Amplitude continued rising to 9.1 mm/s. Alarm was logged by the "
        "online system. No corrective work order was raised in response to the Zone D alarm.", BODY))
    make_pdf("VS-204_vibration_trend_log.pdf", f)


# -------- DOC 4: Incident Report INC-2024-07 (THE FAILURE) --------
def doc_incident_inc0724():
    f = []
    f.append(Paragraph("INCIDENT REPORT", H))
    f.append(Paragraph("Document Ref: INC-2024-07  |  Equipment Failure / Unplanned Downtime", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Incident Date", FAIL_DATE.strftime(D)],
        ["Asset", "Pump P-204 (Boiler Feed Water)"],
        ["Event", "Catastrophic DE bearing seizure"],
        ["Downtime", "38 hours unplanned"],
        ["Reported By", "Shift Engineer, Unit 2"],
        ["Severity", "High - production loss, no injury"],
    ]))
    f.append(Spacer(1, 10))
    f.append(Paragraph("Event Description", H2))
    f.append(Paragraph(
        "Pump P-204 drive-end bearing seized during normal operation at 14:20. Pump tripped on high "
        "vibration. Inspection confirmed complete failure of the DE bearing (SKF 6316) with secondary "
        "shaft scoring. Boiler feed water supply switched to standby pump P-205. Unit 2 load reduced "
        "for 38 hours during repair.", BODY))
    f.append(Paragraph("Preliminary Cause", H2))
    f.append(Paragraph(
        "Bearing failure due to progressive wear that was detected but not corrected in time. "
        "Root cause analysis to reference inspection and maintenance history for P-204.", BODY))
    f.append(Paragraph("Cross-References", H2))
    f.append(Paragraph(
        "This incident should be reviewed against the maintenance and inspection history of P-204. "
        "Relevant records exist in the plant document systems but were not consolidated prior to failure.", BODY))
    make_pdf("INC-2024-07_incident_report.pdf", f)


# -------- DOC 5: Procedure M-118 (WHAT SHOULD HAVE HAPPENED) --------
def doc_procedure_m118():
    f = []
    f.append(Paragraph("STANDARD MAINTENANCE PROCEDURE", H))
    f.append(Paragraph("Document Ref: M-118  |  Rotating Equipment Condition Monitoring", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Procedure ID", "M-118"],
        ["Title", "Condition Monitoring & Bearing Management, Rotating Equipment"],
        ["Applies To", "Centrifugal pumps, motors, fans (rotating assets)"],
        ["Governing Regulation", "OISD-132 (Inspection of Rotating Equipment)"],
        ["Revision", "Rev 4"],
    ]))
    f.append(Spacer(1, 10))
    f.append(Paragraph("Clause 4.3 - Elevated Vibration Response", H2))
    f.append(Paragraph(
        "Where a bearing is flagged with elevated vibration (ISO 10816-3 Zone C) during inspection, "
        "the bearing shall be scheduled for replacement within 14 days of the inspection date. "
        "Interim lubrication may be applied but does not substitute for replacement. Deferral beyond "
        "14 days requires documented risk assessment and area manager approval, not supervisor "
        "sign-off alone.", BODY))
    f.append(Paragraph("Clause 4.4 - Zone D Alarm Response", H2))
    f.append(Paragraph(
        "Any asset crossing the Zone D alarm threshold (7.1 mm/s RMS for this class) requires an "
        "immediate corrective work order and consideration for controlled shutdown. Online alarms "
        "shall not be left without a raised work order.", BODY))
    f.append(Paragraph("Regulatory Basis", H2))
    f.append(Paragraph(
        "This procedure implements the inspection and integrity requirements of OISD-132. "
        "Non-compliance with bearing replacement timelines constitutes a procedural deviation "
        "reportable under the plant integrity management system.", BODY))
    make_pdf("M-118_procedure.pdf", f)


# ============================================================
# DISTRACTOR / VOLUME DOCS (realistic corpus, no chain)
# ============================================================
def doc_other_pump_manual():
    f = []
    f.append(Paragraph("PUMP OPERATING MANUAL (EXTRACT)", H))
    f.append(Paragraph("Asset: Centrifugal Pump P-210  |  Cooling Water Service", META))
    f.append(Spacer(1, 8))
    f.append(Paragraph("Bearing Lubrication", H2))
    f.append(Paragraph(
        "P-210 uses grease-lubricated deep-groove ball bearings. Recommended relubrication interval "
        "is 2000 operating hours with ISO VG 68 grade grease. Over-lubrication can cause elevated "
        "bearing temperature. Normal DE bearing vibration for this unit is below 2.8 mm/s RMS.", BODY))
    f.append(Paragraph("Routine Checks", H2))
    f.append(Paragraph(
        "Weekly: check seal leakage, bearing temperature, and suction pressure. Monthly: vibration "
        "spot reading. P-210 is not on continuous online monitoring.", BODY))
    make_pdf("P-210_pump_manual_extract.pdf", f)


def doc_seal_procedure():
    f = []
    f.append(Paragraph("MECHANICAL SEAL REPLACEMENT PROCEDURE", H))
    f.append(Paragraph("Document Ref: M-142  |  Pump Sealing Systems", META))
    f.append(Spacer(1, 8))
    f.append(Paragraph("Scope", H2))
    f.append(Paragraph(
        "Covers replacement of mechanical seals on centrifugal pumps. Not applicable to bearing "
        "maintenance (see M-118 for bearing and condition monitoring). Seal faces must be inspected "
        "for scoring before reassembly.", BODY))
    f.append(Paragraph("Torque Values", H2))
    seal = [["Fastener", "Torque (Nm)"], ["Gland bolts", "45"], ["Cover bolts", "70"], ["Coupling", "120"]]
    t = Table(seal, colWidths=[2.5*inch, 2.0*inch])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2F5496")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    f.append(t)
    make_pdf("M-142_seal_procedure.pdf", f)


def doc_motor_inspection():
    f = []
    f.append(Paragraph("MOTOR INSPECTION REPORT", H))
    f.append(Paragraph("Document Ref: IR-560  |  Electric Motor M-204 (drives Pump P-204)", META))
    f.append(Spacer(1, 8))
    f.append(kv_table([
        ["Inspection Date", (IR_DATE + timedelta(days=2)).strftime(D)],
        ["Asset", "Electric Motor M-204 (drives P-204)"],
        ["Inspector", "R. Krishnan"],
    ]))
    f.append(Spacer(1, 8))
    f.append(Paragraph(
        "Motor M-204 insulation resistance and winding temperature within normal limits. Motor DE and "
        "NDE bearing vibration nominal (below 3.0 mm/s). No action required on the motor. Note: the "
        "driven pump P-204 has a separate elevated-vibration finding recorded under IR-556.", BODY))
    make_pdf("IR-560_motor_inspection.pdf", f)


def doc_lube_spec():
    f = []
    f.append(Paragraph("LUBRICATION SPECIFICATION SHEET", H))
    f.append(Paragraph("Document Ref: LS-07  |  Rotating Equipment Lubricants", META))
    f.append(Spacer(1, 8))
    lube = [
        ["Equipment Class", "Lubricant Grade", "Interval"],
        ["Feed water pumps", "ISO VG 68", "1500 hrs"],
        ["Cooling water pumps", "ISO VG 68 grease", "2000 hrs"],
        ["Fans", "ISO VG 46", "3000 hrs"],
    ]
    t = Table(lube, colWidths=[2.2*inch, 2.0*inch, 1.6*inch])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    f.append(t)
    f.append(Spacer(1, 8))
    f.append(Paragraph(
        "Feed water pumps (including P-204) use ISO VG 68 oil, relubrication interval 1500 operating "
        "hours. Do not exceed fill line. Lubrication is an interim measure only and does not extend "
        "bearing life where mechanical wear is present.", BODY))
    make_pdf("LS-07_lubrication_spec.pdf", f)


def doc_shift_handover():
    f = []
    f.append(Paragraph("SHIFT HANDOVER LOG", H))
    f.append(Paragraph("Unit 2  |  Rotating Equipment Watch", META))
    f.append(Spacer(1, 8))
    f.append(Paragraph(
        "Handover " + (VS_DATE + timedelta(days=1)).strftime(D) + ": P-204 vibration alarm annunciated "
        "on the online system overnight. Noted in the alarm list. Standby pump P-205 confirmed "
        "available. No work order raised this shift - flagged for day team to review P-204 alarm. "
        "P-210 and P-201 normal.", BODY))
    make_pdf("SH-0717_shift_handover.pdf", f)


def doc_generic_safety_sop():
    f = []
    f.append(Paragraph("SAFETY STANDARD OPERATING PROCEDURE", H))
    f.append(Paragraph("Document Ref: SOP-09  |  Pump House Confined Areas", META))
    f.append(Spacer(1, 8))
    f.append(Paragraph(
        "General safety SOP for pump house operations. Covers PPE, lockout-tagout for pump maintenance, "
        "and confined space entry near sumps. Any pump isolation for bearing or seal work requires a "
        "work permit and LOTO per this SOP. This document does not cover condition monitoring "
        "thresholds (see M-118).", BODY))
    make_pdf("SOP-09_pumphouse_safety.pdf", f)


if __name__ == "__main__":
    doc_inspection_ir556()
    doc_maintlog_ml1183()
    doc_sensorlog_vs204()
    doc_incident_inc0724()
    doc_procedure_m118()
    doc_other_pump_manual()
    doc_seal_procedure()
    doc_motor_inspection()
    doc_lube_spec()
    doc_generic_safety_sop()
    doc_shift_handover()
    files = sorted(os.listdir(OUT))
    print(f"Generated {len(files)} synthetic PDFs:")
    for x in files:
        print("  -", x)
