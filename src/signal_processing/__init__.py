"""Signal processing module — Stage 1~7 pipeline for single differential pressure sensor.

Implements the 7-Stage signal processing pipeline defined in
`3.개발기술/03_신호처리_사양서.md` v0.4 §0.6:

  Stage 1: Sensor → MCU (I²C raw acquisition) — out of scope (hardware)
  Stage 2: Raw pressure refinement (HPF·LPF·Notch + 4-branch BPF)  → filters.py
  Stage 3: Flow derivation (Phase 0 blower-inverse / Phase 1 pneumotacho) → flow_estimator.py
  Stage 4: Breath pattern analysis (TV·RR·MV·IT/ET·Leak)             → breath_analyzer.py
  Stage 5: Event detection (Apnea·Hypopnea·CSR per AASM 2023)        → event_detector.py
  Stage 6: AI·XAI analysis (Cardiogenic·FOT R/X classification)      → event_detector.py
  Stage 7: Display·Storage·Transmission                              → demo / downstream

Synthetic signal generator (Stage 0) for in-vitro PoC and unit testing:
                                                                       → generator.py

References
----------
* 03_신호처리_사양서.md v0.4 §0.6 (Stage View)
* AASM Manual for the Scoring of Sleep v2.6 (2023)
* Berry RB et al. J Clin Sleep Med 2012;8(5):597-619
* Ayappa I et al. Chest 1999;116(3):660-6 (Cardiogenic Oscillation)
* Farré R et al. Am J Respir Crit Care Med 1999;160(5):1810-5 (FOT)
"""
from __future__ import annotations

from . import breath_analyzer, event_detector, filters, flow_estimator, generator

__all__ = [
    "generator",
    "filters",
    "flow_estimator",
    "breath_analyzer",
    "event_detector",
]
