# Data availability

The source data required by this project is not distributed with the
repository and is not licensed under the MIT License. It must not be committed
or redistributed.

Authorized users may place the following files in their local `data/`
directory:

- `combined_training_data.csv`
- `co_ind.csv`
- `material_properties.md`

The `data/` directory is ignored by Git. The public test suite uses generated
synthetic data and does not require access to the private source data.

The derived grouped-split manifest at
`data/splits/baseline_split_seed42.json` is checked in. It contains operating
condition assignments and source checksums, but no AEPS fields, element-point
coordinates, or material-property values.
