# Yet Another PTX Designer

This is an MRI parallel transmit (pTx) pulse designer. The original purpose is have something independent of the vendors' frameworks, so it is more convenient to test and run new algorithms. YAPtxD is written and currently maintained by Minghao Zhang.

## Functionality and usages

Existing functions focus on 2D small-tip-angle designs, including CLS and MLS, for shimming and spokes form. The field maps and optimizers are structured to allow multi-slice, multi-band designs.

Bayesian Optimization of GrAdient Trajectory (BOGAT) is implemented for spokes form. When using BOGAT, please cite [my paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30007).

Examples of [shimming](shim_test.py) and [multi-slice BOGAT design](multislice-bogat-test.py) are included to demonstrate usage.

## Contributions

The intention of this project is to include as many new methods as possible, so contribution is very welcome. I would recommend getting in touch before divingi in, because at present YAPtxD may undergo significant code restructuring.

## License

This project uses [Available Source License](LICENSE), which is an academic **non-commercial** variant of GPL.

Please note that the BOGAT functionality is the subject of a UK patent.