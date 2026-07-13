---
status: accepted
---

# Evaluate the baseline on interpolation only

The data-driven DeepONet baseline will be evaluated only on interpolation to unseen operating conditions within the sampled domain. Extrapolation and special handling of the high-AEPS operating region are explicitly outside the baseline's scope. This keeps the baseline focused on a controlled reference result.

The runtime follows the same boundary: it returns the 48-point AEPS field only for the canonical element points and for operating conditions inside the inclusive sampled envelope. Unsupported coordinates or operating conditions are rejected rather than clamped or extrapolated. The exact runtime rules are recorded in the [baseline inference contract](../baseline-inference-contract.md).

## Primary split protocol

The 117 unique temperature-amplitude groups are partitioned deterministically into 82 training groups, 17 validation groups, and 18 locked test groups, corresponding to 246, 51, and 54 simulation cases. All three PCB-modulus variants of a temperature-amplitude pair stay together. The base grid and enriched high-temperature/high-amplitude region are stratified across partitions, and every validation or test temperature and amplitude level must also occur in training. The same saved split manifest is used for every baseline training seed.

The 99-group base-grid stratum is the Cartesian product of these numerically ordered levels:

- temperatures: `[-40, -19.25, 1.5, 22.5, 42, 62.75, 83.5, 104.25, 115, 120, 125]` degrees Celsius;
- vibration displacement amplitudes: `[0.2, 0.29, 0.375, 0.46, 0.55, 0.64, 0.725, 0.81, 0.9]` millimetres.

The other 18 temperature-amplitude groups form the enrichment stratum. Within each stratum, groups are sorted lexicographically by ascending `(temperature, amplitude)`.

Manifest generation creates exactly one NumPy `Generator(PCG64(42))`. It permutes the sorted base-grid stratum first and then permutes the sorted enrichment stratum using the same advancing generator state. The first `69`, next `15`, and final `15` permuted base-grid groups are assigned to training, validation, and test respectively. The first `13`, next `2`, and final `3` permuted enrichment groups are assigned in the same order. The accepted result satisfies the coverage rule.

Each selected group is expanded to its three PCB-modulus cases in ascending modulus order `[20, 23.5, 27]` gigapascals. The complete case assignments and generation metadata are saved as the authoritative split manifest. Once saved, that manifest is never regenerated for a different model or training seed.

## Authoritative manifest artifact

The manifest is one checked-in JSON file at `data/splits/baseline_split_seed42.json`. It records a schema version, the three canonical source-file SHA-256 checksums, and the complete split-generation metadata: bit generator, split seed, stratum definitions, sort order, generator-call order, allocation slices, and expansion rule.

The manifest contains all 351 expanded simulation cases. Each case records its one-based source-data row excluding the CSV header, temperature, vibration displacement amplitude, PCB Young's modulus, stratum, and assigned partition. Cases have one canonical serialized order: training, validation, then test; within each partition they retain the accepted permuted-group order, and the three cases within a group use ascending PCB-modulus order.

The manifest identity is the SHA-256 checksum of the finished file bytes. That identity is stored in every dependent run configuration, preprocessing artifact, checkpoint, prediction artifact, and report; it is not embedded recursively inside the manifest itself.
