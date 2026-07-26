# Third-party notices

Software / source code in this project is released under the MIT License
(see [LICENSE](LICENSE)). The bundled paper PDF is under CC BY 4.0 — see
[NOTICE](NOTICE). The notices below cover third-party packages declared in
`requirements.txt`. Those packages are not vendored here; install them
separately and consult the license text shipped with each package for
authoritative terms.

| Package | Typical license | Role | Project URL |
| --- | --- | --- | --- |
| [streamlit](https://pypi.org/project/streamlit/) | Apache-2.0 | Interactive web app (`app.py`) | https://streamlit.io/ |
| [numpy](https://pypi.org/project/numpy/) | BSD-3-Clause (and related permissive notices in the distribution) | Numerical arrays / results | https://numpy.org/ |
| [pandas](https://pypi.org/project/pandas/) | BSD-3-Clause | Data handling | https://pandas.pydata.org/ |
| [tensornetwork](https://pypi.org/project/tensornetwork/) | Apache-2.0 | Tensor-network backend | https://github.com/google/TensorNetwork |
| [matplotlib](https://pypi.org/project/matplotlib/) | PSF-based (matplotlib license) | Plotting in `task_scheduling_tests.ipynb` | https://matplotlib.org/ |

Version constraints are listed in `requirements.txt`. Transitive dependencies
pulled in by the packages above remain under their own upstream licenses.
