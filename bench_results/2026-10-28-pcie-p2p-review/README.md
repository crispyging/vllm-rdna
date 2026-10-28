# PR23 review receipts

The companion `docs/rdna2/pcie-p2p-kv-rework.md` defines the required
Flash-Next hybrid rework and outstanding qualification.

`original-pr23-failures.json` records three reproduced failures in original
head bbf8c76bae1885c441576527e92dc41e5c869175. To rerun, copy the archived
`pcie_p2p_connector.py.txt`, `pcie_p2p_layout.py.txt` and `reproduce.py.txt`
into a scratch directory, remove their `.txt` suffixes, then run
`.venv/bin/python /path/to/scratch/reproduce.py`. The probe executes the
original scheduler/worker class bodies with explicit CPU stubs for transport;
it does not import the full vLLM runtime or claim to exercise HIP.

`pr23-hybrid-tests.log.txt` records 6 geometry and 7 recurrent-handoff tests
passing in the existing isolated ROCm environment. Transport is mocked in
these tests; real transfers and serving remain unqualified.

The source/build script archives document the private dependency feasibility
experiment, not a production installer. UCX/NIXL revisions are pinned to the
repository Dockerfile. No system packages, shared venvs or serving extensions
are changed by this experiment. Build status is recorded in the review doc.
