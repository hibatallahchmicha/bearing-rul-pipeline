"""Bearing fault frequencies, derived from geometry -- not looked up from a table.

A rolling-element bearing fails at one of four locations, and each imprints
a distinct, physically-derived frequency onto the vibration signal:

- Outer race defect  -> BPFO (Ball Pass Frequency, Outer race)
- Inner race defect  -> BPFI (Ball Pass Frequency, Inner race)
- Roller/ball defect -> BSF  (Ball Spin Frequency)
- Cage defect        -> FTF  (Fundamental Train Frequency)

Standard rolling-element kinematics, assuming pure rolling with no slip
(Harris, *Rolling Bearing Analysis*, 5th ed., ch. 20). Computed from
geometry + shaft speed rather than hardcoded, so features_freq.py isn't
silently tied to one bearing model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Physical dimensions of a rolling-element bearing.

    Defaults are the Rexnord ZA-2115 double-row bearing used in the NASA
    IMS dataset (geometry as published in Qiu et al., 2006, Journal of
    Sound and Vibration).
    """

    n_rolling_elements: int = 16
    roller_diameter_mm: float = 8.4074      # d
    pitch_diameter_mm: float = 71.501       # D
    contact_angle_deg: float = 15.17


@dataclass(frozen=True)
class FaultFrequencies:
    bpfo_hz: float
    bpfi_hz: float
    bsf_hz: float
    ftf_hz: float


def compute_fault_frequencies(
    shaft_speed_rpm: float, geometry: BearingGeometry = BearingGeometry()
) -> FaultFrequencies:
    """Derive BPFO/BPFI/BSF/FTF from shaft speed and bearing geometry.

        BPFO = (n/2) * fr * (1 - (d/D) cos(theta))
        BPFI = (n/2) * fr * (1 + (d/D) cos(theta))
        BSF  = (D / 2d) * fr * (1 - ((d/D) cos(theta))^2)
        FTF  = (fr/2) * (1 - (d/D) cos(theta))

    where fr is shaft rotation frequency in Hz (RPM / 60).
    """
    fr = shaft_speed_rpm / 60.0
    n = geometry.n_rolling_elements
    ratio = (geometry.roller_diameter_mm / geometry.pitch_diameter_mm) * math.cos(
        math.radians(geometry.contact_angle_deg)
    )

    return FaultFrequencies(
        bpfo_hz=(n / 2) * fr * (1 - ratio),
        bpfi_hz=(n / 2) * fr * (1 + ratio),
        bsf_hz=(geometry.pitch_diameter_mm / (2 * geometry.roller_diameter_mm)) * fr * (1 - ratio**2),
        ftf_hz=(fr / 2) * (1 - ratio),
    )