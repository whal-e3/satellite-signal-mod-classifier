# Overview

The model used for testing PE is "backbone-freeze" SNR 10~20db model.

## 1. Technical + Visual mapping
prompt_prefix = (
    "This signal is shown in four views: constellation, time-domain, frequency-domain, and waterfall. "
    "Possible modulations include BPSK, QPSK, APSK, FM, FSK, GMSK, CW, or CSS. This signal likely uses"
)

## 2. Observation first, human like
prompt_prefix = (
    "These four signal plots help determine modulation. Types include phase-based (BPSK, QPSK), "
    "frequency-based (FM, FSK, GFSK), APSK, and CSS. This signal most likely uses"
)

## 3. Use-case or application context
prompt_prefix = (
    "Captured signal shown as constellation, time, frequency, and waterfall diagrams. "
    "Possible modulations include digital (BPSK, QPSK), analog (WBFM, NBFM), or spread types (CSS, CW). It uses"
)

## 4. Contrastive explanation
prompt_prefix = (
    "Modulations appear differently in signal plots. BPSK has 2 points, QPSK 4, APSK uses rings, "
    "FM spreads wide, CSS shows chirps. Based on these views, this signal uses"
)

## 5. Scientific / Analysis-focused
prompt_prefix = (
    "Signal analysis identifies modulation from visual patterns. This signal is shown in constellation, time, frequency, and waterfall views. "
    "Possible classes include PSK, APSK, FSK, FM, GMSK, CSS, and CW. Based on its structure, this signal uses"
)