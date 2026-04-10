const articleText = document.getElementById("articleText");
const useLlm = document.getElementById("useLlm");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusLine = document.getElementById("statusLine");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const verdictText = document.getElementById("verdictText");
const summaryText = document.getElementById("summaryText");
const riskText = document.getElementById("riskText");
const referencesList = document.getElementById("referencesList");
const warningsCard = document.getElementById("warningsCard");
const warningsList = document.getElementById("warningsList");

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
    const confPct = Math.round((data.prediction.confidence || 0) * 100);
    const factCheck = data.fact_check || null;
    const verdictLabel = factCheck
      ? (factCheck.has_trusted_references ? factCheck.verdict : `${factCheck.verdict} (fallback-only)`)
      : "N/A";
    const predictionLabel = data.prediction.label_name === "review"
      ? "REVIEW"
      : data.prediction.label_name.toUpperCase();

    predictionText.textContent = predictionLabel;
    confidenceText.textContent = `${confPct}%`;
    confidenceBar.style.width = `${confPct}%`;

    verdictText.textContent = verdictLabel;
    summaryText.textContent = data.summary || "No summary available.";
    riskText.textContent = data.risk_explanation || "No risk explanation available.";

    renderReferences(factCheck);
    const warnings = [...(data.warnings || [])];
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
