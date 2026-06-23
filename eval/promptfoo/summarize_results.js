/**
 * Parses PromptFoo results JSON and prints a clean summary with
 * resume-ready metrics: accuracy, routing accuracy, hallucination rate, latency.
 *
 * Usage: node eval/promptfoo/summarize_results.js
 */

const fs = require("fs");
const path = require("path");

const resultsPath = path.join(__dirname, "results", "latest.json");

if (!fs.existsSync(resultsPath)) {
  console.error("No results found. Run: npx promptfoo eval --config eval/promptfoo/promptfooconfig.yaml");
  process.exit(1);
}

const results = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
const tests = results.results?.results ?? [];

if (tests.length === 0) {
  console.error("Results file is empty or malformed.");
  process.exit(1);
}

// ── aggregate metrics ──────────────────────────────────────────────────────

let totalTests = tests.length;
let passed = 0;
let routingCorrect = 0;
let routingTotal = 0;
let hallucinations = 0;
let hallucinationTotal = 0;
let latencies = [];
let confidences = [];

for (const test of tests) {
  const allAssertions = test.gradingResult?.componentResults ?? [];
  const testPassed = test.gradingResult?.pass ?? false;
  if (testPassed) passed++;

  for (const assertion of allAssertions) {
    const reason = assertion.reason ?? "";
    // routing assertion
    if (reason.includes("Correctly routed") || reason.includes("Wrong agent") || reason.includes("expected_agent")) {
      routingTotal++;
      if (assertion.pass) routingCorrect++;
    }
    // hallucination assertion
    if (reason.includes("hallucination") || reason.includes("fabricated") || reason.includes("No source")) {
      hallucinationTotal++;
      if (!assertion.pass) hallucinations++;
    }
  }

  // latency and confidence from metadata
  const meta = test.response?.metadata ?? {};
  if (meta.latency_ms != null) latencies.push(meta.latency_ms);
  if (meta.confidence != null) confidences.push(meta.confidence);
}

const accuracy = ((passed / totalTests) * 100).toFixed(1);
const routingAcc = routingTotal > 0 ? ((routingCorrect / routingTotal) * 100).toFixed(1) : "N/A";
const hallRate = hallucinationTotal > 0 ? ((hallucinations / hallucinationTotal) * 100).toFixed(1) : "N/A";

const avgLatency = latencies.length > 0
  ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
  : null;
const p95Latency = latencies.length > 0
  ? Math.round(latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.95)])
  : null;
const avgConf = confidences.length > 0
  ? (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(3)
  : null;

// ── print summary ──────────────────────────────────────────────────────────

console.log("\n╔══════════════════════════════════════════════════╗");
console.log("║         CX BOT — EVAL SUITE SUMMARY             ║");
console.log("╚══════════════════════════════════════════════════╝\n");

console.log(`  Total test cases  : ${totalTests}`);
console.log(`  Passed            : ${passed} / ${totalTests}`);
console.log(`  Overall accuracy  : ${accuracy}%`);
console.log(`  Routing accuracy  : ${routingAcc}%`);
console.log(`  Hallucination rate: ${hallRate}%`);
if (avgLatency) console.log(`  Avg latency       : ${avgLatency}ms`);
if (p95Latency) console.log(`  p95 latency       : ${p95Latency}ms`);
if (avgConf)    console.log(`  Avg confidence    : ${avgConf}`);

console.log("\n── Resume-ready line ─────────────────────────────────");
console.log(`  "Validated with a ${totalTests}-case PromptFoo eval suite: ${accuracy}% accuracy, `);
console.log(`   ${routingAcc}% domain routing accuracy, ${hallRate}% hallucination rate, `);
if (avgLatency) console.log(`   avg ${avgLatency}ms / p95 ${p95Latency}ms latency"`);
console.log("──────────────────────────────────────────────────────\n");

// ── failed tests ───────────────────────────────────────────────────────────

const failed = tests.filter(t => !t.gradingResult?.pass);
if (failed.length > 0) {
  console.log(`\n── Failed tests (${failed.length}) ───────────────────────────────`);
  for (const t of failed) {
    console.log(`  ✗ ${t.description ?? t.vars?.query ?? "unknown"}`);
    const reasons = (t.gradingResult?.componentResults ?? [])
      .filter(a => !a.pass)
      .map(a => `    → ${a.reason}`)
      .join("\n");
    if (reasons) console.log(reasons);
  }
}
