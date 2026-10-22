# Piezoelectric training

Train with:

```sh
mace_run_train --config examples/piezoelectric/config.yml
```

The extended XYZ files must provide periodic cells, `REF_dipole`, integer
`oxn` atom arrays, and a `REF_piezoelectric` info property of shape `(3, 6)`.
Its Voigt columns are `xx, yy, zz, yz, xz, xy` and its units must be
electron/angstrom-squared. Configurations without the tensor are supported:
their piezoelectric contribution is masked automatically.

`pes+dipole+piezo` is available only for `EnergyDipoleMACE` and
`LREnergyDipoleMACE`. It obtains the prediction from the autograd derivative
of the proper dipole with respect to strain, divided by the cell volume.
