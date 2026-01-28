const { mapSignals } = require("./signal_mapper");

const pickSignals = (signals, classification) => {
  const aligned = signals.filter((signal) => signal.aligns === classification);
  const sorted = aligned.sort((a, b) => b.strength - a.strength);

  // Require a minimal strength to avoid weak or noisy explanations.
  return sorted.filter((signal) => signal.strength >= 0.25).slice(0, 2);
};

const buildExplanation = ({ classification, features }) => {
  if (!classification || !features) {
    return { explanation: "Limited feature detail; summary reflects aggregate acoustic statistics." };
  }

  const signals = mapSignals(features);
  const chosen = pickSignals(signals, classification);

  if (!chosen.length) {
    return { explanation: "Feature evidence is mixed; summary reflects aggregate acoustic statistics." };
  }

  const phrases = chosen.map((signal) => signal.phrase);
  const explanation = phrases.length === 1 ? phrases[0] : `${phrases[0]} and ${phrases[1]}`;
  return { explanation };
};

module.exports = {
  buildExplanation,
};
