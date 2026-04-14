const articleText = document.getElementById("articleText");
const useLlm = document.getElementById("useLlm");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusLine = document.getElementById("statusLine");

const agreementBanner = document.getElementById("agreementBanner");
const agreementText = document.getElementById("agreementText");
const mlPatternText = document.getElementById("mlPatternText");
const mlConfidenceText = document.getElementById("mlConfidenceText");
const mlConfidenceBar = document.getElementById("mlConfidenceBar");
const llmVerdictText = document.getElementById("llmVerdictText");
const llmConfidenceText = document.getElementById("llmConfidenceText");
const llmSourceText = document.getElementById("llmSourceText");
const finalVerdictText = document.getElementById("finalVerdictText");
const finalConfidenceText = document.getElementById("finalConfidenceText");
const finalConfidenceBar = document.getElementById("finalConfidenceBar");
const agreementStatusText = document.getElementById("agreementStatusText");
const finalWhyText = document.getElementById("finalWhyText");
const summaryText = document.getElementById("summaryText");
const riskText = document.getElementById("riskText");
const referencesList = document.getElementById("referencesList");
const warningsCard = document.getElementById("warningsCard");
const warningsList = document.getElementById("warningsList");

function setDecisionFlowVisibility(isVisible) {
  agreementBanner.style.display = isVisible ? "block" : "none";
}

function verdictLabel(value) {
  if (!value) return "N/A";
  const key = value.toLowerCase();
  const labels = {
    real: "Likely Real",
    fake: "Likely Fake",
    review: "Needs Review",
    supported: "Supported",
    contradicted: "Contradicted",
    mixed: "Mixed",
    insufficient_evidence: "Insufficient Evidence",
  };
  return labels[key] || value.replaceAll("_", " ");
}

function applyVerdictChipClass(element, verdict) {
  element.classList.remove("chip-good", "chip-bad", "chip-review", "chip-neutral");
  const key = (verdict || "").toLowerCase();
  if (key === "real" || key === "supported") {
    element.classList.add("chip-good");
    return;
  }
  if (key === "fake" || key === "contradicted") {
    element.classList.add("chip-bad");
    return;
  }
  if (key === "review" || key === "mixed" || key === "insufficient_evidence") {
    element.classList.add("chip-review");
    return;
  }
  element.classList.add("chip-neutral");
}

function applyAgreementStyle(status) {
  agreementBanner.classList.remove("agree", "disagree", "ml-only");
  if (status === "disagree") {
    agreementBanner.classList.add("disagree");
    agreementStatusText.textContent = "DISAGREE";
    return;
  }
  if (status === "ml_only") {
    agreementBanner.classList.add("ml-only");
    agreementStatusText.textContent = "ML ONLY";
    return;
  }
  agreementBanner.classList.add("agree");
  agreementStatusText.textContent = "AGREE";
}

function setStatus(text, kind) {
  statusLine.textContent = text;
  statusLine.className = `status ${kind}`;
}

function renderWarnings(warnings) {
  warningsList.innerHTML = "";
  if (!warnings || warnings.length === 0) {
    warningsCard.style.display = "none";
    return;
  }

  warningsCard.style.display = "block";
  warnings.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    warningsList.appendChild(li);
  });
}

function renderReferences(factCheck) {
  referencesList.innerHTML = "";
  if (!factCheck || !factCheck.references || factCheck.references.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No references returned.";
    referencesList.appendChild(li);
    return;
  }

  factCheck.references.forEach((ref) => {
    const li = document.createElement("li");
    const tag = document.createElement("strong");
    const sourceType = ref.source_type || "fallback";
    tag.textContent = `[${sourceType}] `;

    const link = document.createElement("a");
    link.href = ref.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = ref.title || ref.url;

    li.appendChild(tag);
    li.appendChild(link);
    referencesList.appendChild(li);
  });
}

async function runAnalysis() {
  const text = articleText.value.trim();
  const llmEnabled = useLlm.checked;

  setDecisionFlowVisibility(llmEnabled);
  if (text.length < 20) {
    setStatus("Please enter at least 20 characters.", "error");
    return;
  }

  setStatus("Analyzing article...", "loading");
  analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, use_llm: useLlm.checked }),
    });

    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const mlConfPct = Math.round(((data.ml_assessment?.confidence ?? data.prediction.confidence) || 0) * 100);
    const factCheck = data.fact_check || null;
    const llmAssessment = data.llm_assessment || null;
    const finalAssessment = data.final_assessment || null;

    mlPatternText.textContent = verdictLabel(data.ml_assessment?.label_name || data.prediction.label_name);
    applyVerdictChipClass(mlPatternText, data.ml_assessment?.label_name || data.prediction.label_name);
    mlConfidenceText.textContent = `Confidence: ${mlConfPct}%`;
    mlConfidenceBar.style.width = `${mlConfPct}%`;

    if (llmAssessment) {
      llmVerdictText.textContent = verdictLabel(llmAssessment.verdict);
      applyVerdictChipClass(llmVerdictText, llmAssessment.verdict);
      llmConfidenceText.textContent = `Confidence: ${llmAssessment.confidence}%`;
      llmSourceText.textContent = llmAssessment.has_trusted_references
        ? "Evidence quality: trusted sources"
        : "Evidence quality: fallback sources only";
    } else {
      llmVerdictText.textContent = "N/A";
      applyVerdictChipClass(llmVerdictText, "");
      llmConfidenceText.textContent = "Confidence: N/A";
      llmSourceText.textContent = "LLM verification disabled or unavailable";
    }

    if (finalAssessment) {
      finalVerdictText.textContent = verdictLabel(finalAssessment.verdict);
      applyVerdictChipClass(finalVerdictText, finalAssessment.verdict);
      finalConfidenceText.textContent = `Confidence: ${finalAssessment.confidence}%`;
      finalConfidenceBar.style.width = `${finalAssessment.confidence}%`;
      finalWhyText.textContent = finalAssessment.explanation || "No explanation available.";
      agreementText.textContent = finalAssessment.agreement_status === "disagree"
        ? "ML pattern result and LLM evidence check disagree, so this article is flagged for REVIEW."
        : "ML and LLM outputs were fused into one";
      applyAgreementStyle(finalAssessment.agreement_status);
    } else {
      finalVerdictText.textContent = "N/A";
      applyVerdictChipClass(finalVerdictText, "");
      finalConfidenceText.textContent = "Confidence: N/A";
      finalConfidenceBar.style.width = "0%";
      finalWhyText.textContent = "No final decision explanation available.";
      agreementText.textContent = "Run analysis to compare ML patterns with LLM evidence checks.";
      applyAgreementStyle("ml_only");
    }

    summaryText.textContent = data.summary || "No summary available.";
    riskText.textContent = data.risk_explanation || "No risk explanation available.";

    renderReferences(factCheck);
    const warnings = [...(data.warnings || [])];
    if (finalAssessment?.agreement_status === "disagree") {
      const duplicate = "ML and LLM disagree on this article. Final recommendation is REVIEW.";
      const idx = warnings.indexOf(duplicate);
      if (idx !== -1) {
        warnings.splice(idx, 1);
      }
    }
    if (factCheck && !factCheck.has_trusted_references) {
      warnings.unshift("External references are fallback-only, so verdict confidence is limited.");
    }
    renderWarnings(warnings);
    setStatus("Analysis completed.", "success");
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, "error");
  } finally {
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", runAnalysis);
useLlm.addEventListener("change", () => setDecisionFlowVisibility(useLlm.checked));
setDecisionFlowVisibility(useLlm.checked);
