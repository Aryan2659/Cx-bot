const API_BASE = process.env.CXBOT_URL || "http://localhost:8000";

class CxBotProvider {
  constructor(options) {}

  id() { return "cx-bot-provider"; }

  async callApi(prompt, context) {
    const sessionId = `eval_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

    let response;
    try {
      response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query: prompt }),
      });
    } catch (err) {
      return { error: `Network error — is CX Bot running at ${API_BASE}? (${err.message})` };
    }

    if (!response.ok) {
      return { error: `HTTP ${response.status}: ${await response.text()}` };
    }

    const data = await response.json();

    const meta = {
      agent:      data.agent       ?? "unknown",
      confidence: data.confidence  ?? 0,
      low_conf:   data.low_confidence ?? false,
      sources:    data.sources     ?? [],
      latency_ms: data.latency_ms  ?? 0,
      cache_hit:  data.cache_hit   ?? false,
    };

    return {
      output: `${data.answer || ""}\n\n<!--CXMETA:${JSON.stringify(meta)}-->`,
    };
  }
}

module.exports = CxBotProvider;
