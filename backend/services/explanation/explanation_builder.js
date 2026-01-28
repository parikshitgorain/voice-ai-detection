const { mapSignals } = require("./signal_mapper");

const BLACKLIST = [
  "natural energy dynamics",
  "variable voicing transitions",
  "aggregate acoustic statistics",
];

let lastExplanation = null;
let lastSignature = null;

const containsBlacklisted = (phrase) => {
  if (!phrase) return true;
  const lowered = phrase.toLowerCase();
  return BLACKLIST.some((entry) => lowered.includes(entry));
};

const sortSignals = (signals) => {
  return [...signals].sort((a, b) => {
    if (b.strength !== a.strength) return b.strength - a.strength;
    return a.id.localeCompare(b.id);
  });
};

const filterSignals = (signals, minStrength) => {
  return signals.filter(
    (signal) =>
      signal &&
      Number.isFinite(signal.strength) &&
      signal.strength >= minStrength &&
      signal.phrase &&
      !containsBlacklisted(signal.phrase)
  );
};

const bucketConfidence = (confidenceScore) => {
  if (!Number.isFinite(confidenceScore)) return "unknown";
  if (confidenceScore < 0.4) return "low";
  if (confidenceScore < 0.6) return "mid";
  return "high";
};

const buildSignature = (classification, confidenceScore, signals) => {
  const parts = signals.map((signal) => {
    const value = Number.isFinite(signal.value) ? signal.value.toFixed(3) : "na";
    return `${signal.id}:${value}`;
  });
  return [classification, bucketConfidence(confidenceScore), ...parts].join("|");
};

const pickAligned = (signals, aligns, count, offset = 0) => {
  const aligned = filterSignals(signals.filter((signal) => signal.aligns === aligns), 0.3);
  const sorted = sortSignals(aligned);
  if (!sorted.length) return [];
  const start = Math.min(offset, Math.max(0, sorted.length - count));
  return sorted.slice(start, start + count);
};

const pickAmbiguous = (signals) => {
  const aligned = filterSignals(signals.filter((signal) => signal.aligns === "AMBIGUOUS"), 0.25);
  return sortSignals(aligned).slice(0, 1);
};

const buildConfidentExplanation = (classification, signals) => {
  const chosen = pickAligned(signals, classification, 2);
  if (!chosen.length) return null;
  const phrases = chosen.map((signal) => signal.phrase);
  const base = phrases.length === 1 ? phrases[0] : `${phrases[0]} and ${phrases[1]}`;
  if (classification === "AI_GENERATED") {
    return { text: `${base}, indicating planned speech control.`, used: chosen };
  }
  return { text: `${base}, indicating lack of synthetic consistency.`, used: chosen };
};

const buildAmbiguousExplanation = (signals) => {
  const human = pickAligned(signals, "HUMAN", 1);
  const ai = pickAligned(signals, "AI_GENERATED", 1);
  const ambiguous = pickAmbiguous(signals);

  let sentence = "Signals were mixed";
  const used = [];
  if (human.length && ai.length) {
    sentence += `: ${human[0].phrase}, while ${ai[0].phrase}.`;
    used.push(human[0], ai[0]);
  } else if (human.length || ai.length) {
    const pick = human.length ? human[0] : ai[0];
    sentence += `: ${pick.phrase}.`;
    used.push(pick);
  } else if (ambiguous.length) {
    sentence += `; ${ambiguous[0].phrase}.`;
    used.push(ambiguous[0]);
  } else {
    sentence += ".";
  }

  if (ambiguous.length && !used.some((signal) => signal.id === ambiguous[0].id)) {
    return { text: `${sentence} ${ambiguous[0].phrase}.`, used: [...used, ambiguous[0]] };
  }
  return { text: sentence, used };
};

const buildExplanation = ({ classification, confidenceScore, features }) => {
  if (!classification || !features) {
    return { explanation: "Signals were mixed; insufficient evidence available." };
  }

  const signals = mapSignals(features);
  const isLowConfidence = Number.isFinite(confidenceScore) && confidenceScore < 0.4;
  let result = isLowConfidence
    ? buildAmbiguousExplanation(signals)
    : buildConfidentExplanation(classification, signals);

  if (!result) {
    result = buildAmbiguousExplanation(signals);
    if (!result) {
      result = { text: "Signals were mixed; insufficient evidence available.", used: [] };
    }
  }

  let explanation = result.text;
  let usedSignals = result.used || [];
  let signature = buildSignature(classification, confidenceScore, usedSignals);
  if (explanation === lastExplanation && signature !== lastSignature) {
    if (!isLowConfidence) {
      const alt = pickAligned(signals, classification, 2, 1);
      if (alt.length) {
        const phrases = alt.map((signal) => signal.phrase);
        const base = phrases.length === 1 ? phrases[0] : `${phrases[0]} and ${phrases[1]}`;
        explanation =
          classification === "AI_GENERATED"
            ? `${base}, indicating planned speech control.`
            : `${base}, indicating lack of synthetic consistency.`;
        usedSignals = alt;
        signature = buildSignature(classification, confidenceScore, usedSignals);
      }
    } else {
      const altHuman = pickAligned(signals, "HUMAN", 1, 1);
      const altAi = pickAligned(signals, "AI_GENERATED", 1, 1);
      if (altHuman.length || altAi.length) {
        const humanPhrase = altHuman.length ? altHuman[0].phrase : null;
        const aiPhrase = altAi.length ? altAi[0].phrase : null;
        if (humanPhrase && aiPhrase) {
          explanation = `Signals were mixed: ${humanPhrase}, while ${aiPhrase}.`;
        } else if (humanPhrase || aiPhrase) {
          explanation = `Signals were mixed: ${humanPhrase || aiPhrase}.`;
        }
        usedSignals = [
          ...(altHuman.length ? altHuman : []),
          ...(altAi.length ? altAi : []),
        ];
        signature = buildSignature(classification, confidenceScore, usedSignals);
      }
    }
  }

  if (explanation === lastExplanation && signature !== lastSignature) {
    const fired = filterSignals(signals, 0.3);
    const extra = fired.find(
      (signal) => !usedSignals.some((used) => used.id === signal.id)
    );
    if (extra) {
      explanation = `${explanation} ${extra.phrase}.`;
      usedSignals = [...usedSignals, extra];
      signature = buildSignature(classification, confidenceScore, usedSignals);
    }
  }

  lastExplanation = explanation;
  lastSignature = signature;
  return { explanation };
};

module.exports = {
  buildExplanation,
};
