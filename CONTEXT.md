# Solder-Ball AEPS Surrogate

This context describes the relationship between prescribed PCB loading and material conditions and the accumulated equivalent plastic strain field over fixed solder-element locations.

## Language

**Simulation case**:
One Ansys evaluation for a single operating condition, producing one AEPS field.
_Avoid_: Row, element sample

**Operating condition**:
The combination of temperature, vibration displacement amplitude, and PCB Young's modulus that defines a simulation case.
_Avoid_: Macro features, case parameters

**Vibration displacement amplitude**:
The peak displacement from equilibrium prescribed for the vibration load, measured in millimetres.
_Avoid_: Vibration amplitude, peak-to-peak displacement, RMS displacement

**Element point**:
A fixed two-dimensional `(x, z)` location in millimetres associated with one solder element at which AEPS is reported.
_Avoid_: Sensor, node

**Element index**:
A stable one-based position from 1 through 48 that links an element point to its AEPS value within an AEPS field.
_Avoid_: Ansys element ID, CSV row number

**AEPS field**:
The dimensionless accumulated equivalent plastic strain values over all element points for one simulation case.
_Avoid_: Label vector, output features

**AEPS hotspot**:
The three element points with the largest AEPS values within one AEPS field, selected as `ceil(0.05 × 48)`. They form a three-point discrete approximation to the upper five percent and comprise 6.25% of the 48 element points.
_Avoid_: High-AEPS operating condition, hotspot case

### Physics-informed modeling

**Mechanics residual**:
A quantitative measure of how far a predicted state violates a specified governing mechanics relation, such as equilibrium, kinematic compatibility, constitutive behavior, or a boundary condition.
_Avoid_: Physical constraint, physics prior

**Physics loss**:
A training-objective term assembled from one or more mechanics residuals.
_Avoid_: Physical constraint, constraint loss

**Physics-guided regularizer**:
A penalty expressing expected qualitative behavior without directly evaluating a governing mechanics relation.
_Avoid_: Weak prior, soft constraint, mechanics residual

**Hard enforcement**:
Exact satisfaction of a specified relation by construction rather than by penalizing its violation.
_Avoid_: Hard penalty

**Soft enforcement**:
Approximate satisfaction encouraged through a weighted loss, without a guarantee that the relation holds exactly.
_Avoid_: Soft constraint

**Physics-informed neural operator (PINO)**:
A neural operator that enforces at least one governing mechanics relation through a mechanics residual or hard enforcement.
_Avoid_: Physics-guided neural operator

**Physics-guided neural operator**:
A neural operator that uses physics-guided regularizers but neither mechanics residuals nor hard enforcement of a governing mechanics relation.
_Avoid_: PINO
