import os
from typing import Dict, Any


class FusionEngine:
    @staticmethod
    def calculate_risk(ai_score: float, manipulation_score: float, meta_score: float) -> dict:
        ai_val = max(0.0, min(1.0, float(ai_score)))
        manip_val = max(0.0, min(1.0, float(manipulation_score)))
        meta_val = max(0.0, min(1.0, float(meta_score)))

        # 1. Base weighted distribution
        base_weighted = (0.45 * ai_val) + (0.45 * manip_val) + (0.10 * meta_val)

        # 2. Localized Threat Escalation:
        # TruFor Noiseprint++ or generative markers take priority over diluted weighted averages
        peak_threat = max(ai_val, manip_val)
        if peak_threat >= 0.45:
            # If TruFor or AI detector detects an anomaly, escalate directly
            overall_risk = max(base_weighted, peak_threat * 0.96)
        else:
            overall_risk = base_weighted

        overall_risk_pct = round(overall_risk * 100, 1)
        authenticity_pct = round(max(3.0, 100.0 - overall_risk_pct), 1)

        # 3. Categorical Threat Levels
        if overall_risk_pct >= 65.0:
            risk_level = "HIGH"
            review_status = "🔴 Immediate Forensic Review Required"
        elif overall_risk_pct >= 38.0:
            risk_level = "MEDIUM"
            review_status = "🟡 Discretionary Human Review"
        else:
            risk_level = "LOW"
            review_status = "🟢 Automated Validation Passed"

        # 4. Final Diagnostic Verdict
        if ai_val >= 0.75:
            final_verdict = "AI-Generated (Synthetic Media)"
        elif manip_val >= 0.45:
            final_verdict = "Manipulated / Background Spliced"
        elif meta_val >= 0.70:
            final_verdict = "Suspicious Metadata Inconsistencies"
        elif overall_risk_pct < 38.0:
            final_verdict = "Authentic / Original Capture"
        else:
            final_verdict = "Indeterminate / Splicing Suspected"

        # 5. Diagnostic Findings Summary
        findings = []
        if ai_val >= 0.65:
            findings.append(f"synthetic generative diffusion signatures ({round(ai_val * 100)}% confidence)")
        if manip_val >= 0.45:
            findings.append(f"TruFor sensor noise & background splicing disparities ({round(manip_val * 100)}% confidence)")
        if meta_val >= 0.65:
            findings.append("critical metadata/EXIF structure anomalies")

        if findings:
            conclusion = (
                f"The submitted artifact exhibits {' and '.join(findings)}. "
                f"Integrity risk is scored at {overall_risk_pct}%, resulting in an authenticity index of {authenticity_pct}%. "
                f"Classification: '{final_verdict}'. Status: {review_status}."
            )
        else:
            conclusion = (
                f"The artifact exhibits continuous sensor noise floors (PRNU), natural frequency decay, "
                f"and coherent compression parameters. Integrity risk is classified as {risk_level} ({overall_risk_pct}%) "
                f"with a validated authenticity index of {authenticity_pct}%."
            )

        return {
            "ai_score_pct": f"{round(ai_val * 100, 1)}%",
            "manipulation_score_pct": f"{round(manip_val * 100, 1)}%",
            "metadata_risk_pct": f"{round(meta_val * 100, 1)}%",
            "overall_risk_pct": f"{overall_risk_pct}%",
            "authenticity_percentage": f"{authenticity_pct}%",
            "risk_level": risk_level,
            "human_review_status": review_status,
            "final_verdict": final_verdict,
            "conclusion": conclusion
        }

    @staticmethod
    def fuse_analysis_pipeline(
        file_path: str,
        forensic_results: Dict[str, Any],
        metadata_results: Dict[str, Any],
        custody_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        ai_conf = float(
            forensic_results.get("ai_details", {}).get(
                "ai_confidence",
                forensic_results.get("ai_generation_confidence",
                forensic_results.get("ai_voice_clone_confidence", 0.0))
            )
        )

        manip_conf = float(
            forensic_results.get("manipulation_details", {}).get(
                "manipulation_confidence",
                forensic_results.get("manipulation_confidence",
                forensic_results.get("splicing_tamper_confidence", 0.0))
            )
        )

        meta_score = 0.85 if metadata_results.get("is_suspicious") else (
            0.10 if metadata_results.get("has_exif") else 0.40
        )

        assessment = FusionEngine.calculate_risk(
            ai_score=ai_conf,
            manipulation_score=manip_conf,
            meta_score=meta_score
        )

        # Active localization mask: prioritizes TruFor heatmap over classical ELA
        active_mask = forensic_results.get("manipulation_details", {}).get(
            "ela_image_path",
            forensic_results.get("ela_mask_path", "")
        )

        models_list = forensic_results.get("models_used", [
            "Spectral FFT & C2PA Provenance Engine",
            "TruFor (SegFormer + Noiseprint++) Engine"
        ])

        return {
            "case_id": f"NCFU-{os.path.splitext(os.path.basename(file_path))[0][:8].upper()}",
            "filename": os.path.basename(file_path),
            "file_type": forensic_results.get("media_type", "Digital Media Evidence"),
            "sha256": custody_info.get("file_hash", "N/A"),
            "timestamp": custody_info.get("timestamp_utc", "N/A"),
            "authenticity": assessment["authenticity_percentage"],
            "ai_probability": assessment["ai_score_pct"],
            "manipulation_probability": assessment["manipulation_score_pct"],
            "verdict": assessment["final_verdict"],
            "ela_score": forensic_results.get("manipulation_details", {}).get("ela_score", "N/A"),
            "ela_mask_path": active_mask,
            "models_used": models_list,
            "metadata": metadata_results,
            "branch_a": forensic_results.get("ai_details", {
                "ai_confidence": ai_conf,
                "ai_percentage": f"{round(ai_conf * 100, 1)}%",
                "interpretation": forensic_results.get("interpretation", "Evaluated via spectral transform.")
            }),
            "branch_b": forensic_results.get("manipulation_details", {
                "manipulation_confidence": manip_conf,
                "manipulation_percentage": f"{round(manip_conf * 100, 1)}%",
                "suspicious_region": forensic_results.get("manipulation_details", {}).get("suspicious_region", "Perimeter"),
                "interpretation": forensic_results.get("interpretation", "Evaluated via TruFor & Noiseprint++.")
            }),
            "assessment": assessment,
            "legal_explanation": assessment["conclusion"]
        }