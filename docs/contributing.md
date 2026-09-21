> **Status:** draft — do not publish until the GitHub org (`youseftrk`) is confirmed.

# Contributing — open-physical-sim

## Do not publish yet

Placeholder org: `youseftrk`. Do not treat drafts as a public release until the GitHub organization is confirmed.

## Code license

- Contributions are **Apache-2.0** only.
- No copyleft dependencies without an explicit project decision.
- DCO-style expectation: you have the right to submit the change under Apache-2.0.

## Robot / asset contributions

- Demo reference robots (e.g. Unitree H1/G1) require clear redistribution rights before landing in `assets/robots/`.
- Scene fixtures and env packages follow **CC-BY-4.0** default (**CC0** allowed) — see [`data-license.md`](data-license.md).
- Do not vendor closed binaries or proprietary SDKs.

## PR bar

1. `PhysicalSim/empty-v0` reset/step must keep working (MuJoCo headless).
2. Loader changes must preserve Env Commons / Reconstruct / MJCF detection without inventing alternate package roots.
3. Domain randomization must not remesh captured collision geometry.
4. Do not register H1/G1 env ids until empty-v0 is solid and models are licensed in-tree.
5. Clean-room: describe Open Real2Sim on its own terms; no product-clone claims.
6. Marketplace incentive programs are out of scope for v0 docs and UX copy.

## License checklist

- [ ] New code Apache-2.0; deps permissive (Apache/MIT/BSD).
- [ ] Robot URDFs / meshes have an explicit redistribution note.
- [ ] Fixture scenes declare CC-BY-4.0 or CC0 when redistributed as data.
