---
status: accepted
---

# Use minimal independent material features

The baseline uses the following branch inputs: temperature, vibration displacement amplitude, PCB Young's modulus, and the SAC305 solder Young's modulus and Poisson's ratio evaluated at that temperature. Bulk and shear moduli are excluded because they are algebraically derived from the two selected solder properties, while density and thermal-expansion coefficient are documented as constants rather than supplied as zero-variance learned inputs. This boundary defines the baseline's inputs but does not support a claim of generalization to unseen solder materials.
