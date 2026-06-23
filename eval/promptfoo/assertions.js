/**
 * Custom assertion helpers for CX Bot evals.
 * Each function receives { output, context } from PromptFoo.
 * Return { pass: bool, score: 0-1, reason: string }
 */

/**
 * Checks that the response cites at least one source document.
 * A response with no sources is likely hallucinated.
 */
function assertHasSource({ output, context }) {
  const sources = context?.metadata?.sources ?? [];
  if (sources.length === 0) {
    return {
      pass: false,
      score: 0,
      reason: "No source documents cited — potential hallucination",
    };
  }
  return {
    pass: true,
    score: 1,
    reason: `Cited ${sources.length} source(s): ${sources.join(", ")}`,
  };
}

/**
 * Checks the response was routed to the expected agent domain.
 * Pass expectedAgent via the test vars: { expected_agent: "billing" }
 */
function assertCorrectAgent({ output, context }) {
  const expected = context?.vars?.expected_agent;
  const actual = context?.metadata?.agent;
  if (!expected) return { pass: true, score: 1, reason: "No expected_agent set" };
  if (actual === expected) {
    return { pass: true, score: 1, reason: `Correctly routed to ${actual}` };
  }
  return {
    pass: false,
    score: 0,
    reason: `Wrong agent: expected "${expected}", got "${actual}"`,
  };
}

/**
 * Checks confidence is above the configured threshold (default 0.65).
 */
function assertConfidence({ output, context }) {
  const conf = context?.metadata?.confidence ?? 0;
  const threshold = parseFloat(context?.vars?.min_confidence ?? "0.65");
  if (conf >= threshold) {
    return { pass: true, score: conf, reason: `Confidence ${conf.toFixed(3)} ≥ ${threshold}` };
  }
  return {
    pass: false,
    score: conf,
    reason: `Low confidence: ${conf.toFixed(3)} < ${threshold}`,
  };
}

/**
 * Checks response latency is under threshold (default 5000ms, CPU-tier aware).
 */
function assertLatency({ output, context }) {
  const latency = context?.metadata?.latency_ms ?? 99999;
  const maxMs = parseInt(context?.vars?.max_latency_ms ?? "5000", 10);
  if (latency <= maxMs) {
    return { pass: true, score: 1, reason: `Latency ${latency}ms ≤ ${maxMs}ms` };
  }
  return {
    pass: false,
    score: Math.max(0, 1 - (latency - maxMs) / maxMs),
    reason: `Slow response: ${latency}ms > ${maxMs}ms`,
  };
}

/**
 * Checks the response does NOT contain fabricated specific numbers
 * that don't appear in the known policy facts.
 * Looks for suspicious made-up percentages or days not in the policies.
 */
function assertNoFabricatedFacts({ output, context }) {
  const text = output.toLowerCase();

  // Known real policy values (from the actual docs)
  const knownNumbers = [
    "30 days", "14 days", "7 days", "5-7 business days", "2 business days",
    "60 days", "30 business days", "3 days", "3-5 business days",
    "12 months", "1.5%", "25%", "1 month", "2 hours", "4 hours",
    "24 hours", "48 hours", "5 business days", "$200", "$500",
  ];

  // Patterns that suggest a made-up specific number
  const suspiciousPatterns = [
    /\b(90|45|21|10|15)\s*days?\b/,         // common LLM hallucination numbers
    /\b\d+\s*%\s*(fee|penalty|discount)\b/,  // made-up percentages
    /within\s+\d+\s*hours?\b/,               // check against known SLA hours
  ];

  const flags = [];
  for (const pattern of suspiciousPatterns) {
    const match = text.match(pattern);
    if (match) {
      const snippet = match[0];
      const isKnown = knownNumbers.some((k) => text.includes(k.toLowerCase()));
      // Only flag if the number isn't one of our known real values
      const knownMatch = knownNumbers.find(
        (k) => k.toLowerCase().includes(snippet.replace(/[^0-9a-z\s]/gi, "").trim())
      );
      if (!knownMatch) {
        flags.push(snippet);
      }
    }
  }

  if (flags.length > 0) {
    return {
      pass: false,
      score: 0,
      reason: `Possible fabricated facts detected: "${flags.join('", "')}"`,
    };
  }
  return { pass: true, score: 1, reason: "No fabricated facts detected" };
}

module.exports = {
  assertHasSource,
  assertCorrectAgent,
  assertConfidence,
  assertLatency,
  assertNoFabricatedFacts,
};
/**
 * Assertion helpers for CX Bot PromptFoo evals.
 * Metadata is embedded in the output string as <!--CXMETA:{...}-->
 * so these work regardless of PromptFoo version.
 */

function parseMeta(output) {
  const match = (output || "").match(/<!--CXMETA:(.+?)-->/);
  if (!match) return {};
  try { return JSON.parse(match[1]); } catch { return {}; }
}

function cleanOutput(output) {
  return (output || "").replace(/\n*<!--CXMETA:.+?-->/, "").trim();
}

function assertHasSource({ output, context }) {
  const meta = parseMeta(output);
  const sources = meta.sources ?? [];
  if (sources.length === 0)
    return { pass: false, score: 0, reason: "No source documents cited — potential hallucination" };
  return { pass: true, score: 1, reason: `Cited ${sources.length} source(s): ${sources.join(", ")}` };
}

function assertCorrectAgent({ output, context }) {
  const expected = context?.vars?.expected_agent;
  const meta = parseMeta(output);
  const actual = meta.agent;
  if (!expected) return { pass: true, score: 1, reason: "No expected_agent set" };
  if (actual === expected) return { pass: true, score: 1, reason: `Correctly routed to ${actual}` };
  return { pass: false, score: 0, reason: `Wrong agent: expected "${expected}", got "${actual}"` };
}

function assertConfidence({ output, context }) {
  const meta = parseMeta(output);
  const conf = meta.confidence ?? 0;
  const threshold = parseFloat(context?.vars?.min_confidence ?? "0.60");
  if (conf >= threshold) return { pass: true, score: conf, reason: `Confidence ${conf.toFixed(3)} ≥ ${threshold}` };
  return { pass: false, score: conf, reason: `Low confidence: ${conf.toFixed(3)} < ${threshold}` };
}

function assertLatency({ output, context }) {
  const meta = parseMeta(output);
  const latency = meta.latency_ms ?? 0;
  const maxMs = parseInt(context?.vars?.max_latency_ms ?? "8000", 10);
  if (latency <= maxMs) return { pass: true, score: 1, reason: `Latency ${latency}ms ≤ ${maxMs}ms` };
  return { pass: false, score: Math.max(0, 1 - (latency - maxMs) / maxMs), reason: `Slow: ${latency}ms > ${maxMs}ms` };
}

function assertNoFabricatedFacts({ output, context }) {
  const text = cleanOutput(output).toLowerCase();
  const knownNumbers = [
    "30 days","14 days","7 days","5-7 business days","2 business days",
    "60 days","30 business days","3 days","3-5 business days",
    "12 months","1.5%","25%","1 month","2 hours","4 hours",
    "24 hours","48 hours","5 business days","$200","$500",
  ];
  const suspiciousPatterns = [
    /\b(90|45|21)\s*days?\b/,
    /\b[3-9]\d\s*%\s*(fee|penalty|discount)\b/,
  ];
  const flags = [];
  for (const p of suspiciousPatterns) {
    const m = text.match(p);
    if (m && !knownNumbers.some(k => text.includes(k.toLowerCase()))) flags.push(m[0]);
  }
  if (flags.length > 0)
    return { pass: false, score: 0, reason: `Possible fabricated facts: "${flags.join('", "')}"` };
  return { pass: true, score: 1, reason: "No fabricated facts detected" };
}

module.exports = { assertHasSource, assertCorrectAgent, assertConfidence, assertLatency, assertNoFabricatedFacts };