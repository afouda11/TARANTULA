# TARANTULA

**Xray induced dynamics for the experimental frontier**

A dual functional C++ code which can: 

1) **TDSE_XFEL**: Numerical solution of the time-dependent Schrödinger equation (TDSE) to simulate electronic population transfer dynamics in the presence of ultrafast, intense x-rays pulses for an arbitary number of states and decay channels and includes: 
    - Rotating-wave approximation (RWA)
    - Gaussian pulse envelopes with:
        - focal-volume averaging
        - bandwidth averaging
        - AC-Stark shifts
    - Photon energy and pulse intensity scans
    - Decay channel populations dynamics with user provided:
        - Auger-Meitney decay widths
        - photoionization cross-sections
    - 1 or 2 pulse modes configurations with either:
        - orientational averaging 
        - perpendicular polariztion 

2) **SPAWN_DECAY**: Model wave packet evolution over multiple potential energy surfaces (PES) in multi-step Auger-Meitner decay cascades, via novel trajectory basis function decay spawning alorithm:
    - Current implementation prototypes decay spawning algorithm by modelling wave packet evolution across pre-defined PES from *ab initio* molecular dynamics (AIMD) simulations
    - Algorithm propagates trajectory basis functions (TBFs) acroos multiple potential energy surfaces using:
        - Partial Auger rates for each channel that determine the population transer rate
        - User-defined spawning recurrance paramter to resolves the dispersion of the nuclear wavepacket
    - Spawning algorithm uses phase-matching and bond-length mapping for cascade dynamics

---

## Citation

This code was used for these publications, here the full set theoretical and computational details can be found: 

**TDSE_XFEL**

1. **Site-specific generation of excited state wavepackets with high-intensity attosecond x rays**
   A. E. A. Fouda and P. J. Ho,
   *J. Chem. Phys.* (2021) **154**, 224111
   [https://doi.org/10.1063/5.0050891](https://doi.org/10.1063/5.0050891)

2. **Resonant Double-Core Excitations with Ultrafast, Intense Pulses**
   A. E. A. Fouda, D. Koulentianos, L. Young, G. Doumy and P. J. Ho,
   *Mol. Phys.* (2022) e2133749 
   [https://doi.org/10.1080/00268976.2022.2133749](https://doi.org/10.1080/00268976.2022.2133749)

**SPAWN_DECAY** 

1. **Quantum molecular charge-transfer model for multistep Auger–Meitner decay cascade dynamics**
    A. E. A. Fouda, S. H. Southworth and P. J. Ho
    *J. Chem. Theory Comput.* (2024) 20, 20, 8782-8794
    [https://doi.org/10.1021/acs.jctc.4c00778](https://doi.org/10.1021/acs.jctc.4c00778)

If you like the code and/or research, please give em a cite!

## Table of contents for this package doc:

1.  [Code Architecture](#code-architecture)
2.  [Dependencies](#dependencies)
3.  [Build Instructions](#build-instructions)
4.  [Running Simulations](#running-simulations)
5.  [Input File Reference](#input-file-reference)
6.  [Output Files](#output-files)
7.  [Examples](#examples)

*This doc is a work in progress, please reach out to me if you have questions about this code*
---

## Code Architecture

```
TARANTULA/
├── src/
│   ├── main.cpp               # Entry point: dispatches to TDSE_XFEL or SPAWN_DECAY
│   ├── xfel_tdse.cpp          # XFEL_TDSE() — full TDSE setup, time loop, and output
│   ├── pulse_interaction.cpp  # TDSEUTILITY class — focal-volume avg, bandwidth avg, EOM driver
│   ├── rk4.cpp                # EOMDRIVER class — RK4 integrator, REQ (TDSE), REQ_SPAWN, decay
│   ├── spawn_decay.cpp        # SPAWN_DECAY() — spawning algorithm on AIMD data
│   └── read_and_write.cpp     # I/O: read_matrix, file2vector, FILEWRITER class
├── include/
│   ├── xfel_tdse.h            # Declaration of XFEL_TDSE()
│   ├── spawn_decay.h          # Declaration of SPAWN_DECAY()
│   ├── pulse_interaction.h    # TDSEUTILITY class declaration
│   ├── rk4.hpp                # EOMDRIVER class declaration
│   ├── read_and_write.h       # I/O function/class declarations + templates
│   └── vectypedef.hpp         # Type aliases (complexd, vec1x, vec2x, etc.)
├── bin/
│   └── tarantula              # Compiled binary
├── obj/                       # Object files
├── examples/
│   ├── n2o_tdse/              # N2O XFEL TDSE example (publication test case)
│   │   ├── inputs/            # Input data files
│   │   ├── matrix_elements/   # Hamiltonian matrix elements
│   │   ├── outputs/           # Simulation outputs
│   │   └── reference/         # Reference outputs for regression testing
│   └── ibr_spawn/             # IBr Auger-Meitner spawning example (publication test case)
│       ├── inputs/            # Input data files + AIMD trajectories
│       ├── outputs/           # Simulation outputs
│       └── reference/         # Reference outputs for regression testing
├── tests/
│   ├── run_tests.sh           # Master regression test script
│   ├── test_n2o.sh            # Regression test for N₂O example
│   └── test_ibr.sh            # Regression test for IBr example
└── Makefile                   # Build system
```

### Class Summary

| Class | File | Role |
|-------|------|------|
| `TDSEUTILITY` | `pulse_interaction.cpp` | Manages pulse parameters, focal-volume/bandwidth averaging, and drives the EOM time loop for the TDSE mode |
| `EOMDRIVER` | `rk4.cpp` | Implements the RK4 integrator, the equations of motion (`REQ` for TDSE, `REQ_SPAWN` for decay dynamics), population loss functions, Stark shifts, and dipole matrix element computation |
| `FILEWRITER` | `read_and_write.cpp` | Handles all output writing: time-dependent populations, variable scans, spawning populations, and auxiliary data (bond lengths, charges, spins, energies) |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **C++ compiler** | C++11 or later | `g++`, `clang++`, or `icpx` (Intel oneAPI) |
| **Armadillo** | ≥ 9.x | Linear algebra (matrix storage for Hamiltonian) |
| **OpenMP** | (optional) | Parallelisation of focal-volume/bandwidth averaging loops |

### Installing Armadillo

**macOS (Homebrew):**
```bash
brew install armadillo
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libarmadillo-dev
```

**From source:**
```bash
wget https://sourceforge.net/projects/arma/files/armadillo-12.8.0.tar.xz
tar xf armadillo-12.8.0.tar.xz
cd armadillo-12.8.0
cmake . -DCMAKE_INSTALL_PREFIX=$HOME/armadillo-install
make install
```

---

## Build Instructions

### Prerequisites

1. **A C++11 compiler** — the default Makefile uses the Intel C++ compiler (`icpc`).
   If you don't have Intel compilers, switch to `g++` or `clang++` (see below).
2. **Armadillo** — installed somewhere accessible (see [Dependencies](#dependencies)).
3. **OpenMP** — used for parallelisation (comes with `icpc` and `g++`; on macOS with `clang++` you need `libomp`).

### Step 1: Configure the Makefile

Before building, edit the `Makefile` to set two things:

#### Compiler

The default is `icpc` (Intel C++ classic). To change it, uncomment/set `CXX`:

```makefile
# Intel (default in the shipped Makefile):
CXX = icpc

# GNU:
CXX = g++

# Intel oneAPI (newer):
CXX = icpx

# macOS Xcode clang (requires libomp for OpenMP):
CXX = clang++
```

> **Intel compiler setup**: If using Intel compilers on a cluster, source the Intel
> environment first, e.g.:
> ```bash
> source /path/to/intel/oneapi/setvars.sh
> ```

#### Armadillo Include Path

Set the `CFLAGS` line to point to your Armadillo installation:

```makefile
# Custom install path (default in shipped Makefile):
CFLAGS = $(OPTS) /path/to/armadillo-install/include -fopenmp

# System-installed Armadillo (Homebrew / apt — no path needed):
CFLAGS = $(OPTS) -fopenmp
```

> **Note**: If Armadillo is installed via Homebrew (`brew install armadillo`) or
> a system package manager, the include path is already in the compiler's default
> search path, so you can simply use `CFLAGS = $(OPTS) -fopenmp`.

#### macOS with clang++ and Homebrew

If using `clang++` on macOS, you need Homebrew's `libomp` for OpenMP:
```bash
brew install libomp
```
Then update the Makefile:
```makefile
CXX = clang++
CFLAGS = $(OPTS) -I$(shell brew --prefix libomp)/include -fopenmp
LDFLAGS = -L$(shell brew --prefix libomp)/lib -lomp
```

### Step 2: Build

```bash
cd /path/to/TARANTULA
make clean
make
```

The compiled binary is placed at `bin/tarantula`.

### Makefile Overview

The shipped `Makefile` compiles the following source files:

```
read_and_write.cpp  pulse_interaction.cpp  main.cpp  rk4.cpp  xfel_tdse.cpp  spawn_decay.cpp
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CXX` | `icpc` | C++ compiler |
| `OPTS` | `-std=c++11 -O0 -Wall -I` | Compiler standard and optimisation flags |
| `CFLAGS` | `$(OPTS) <armadillo-path> -fopenmp` | Compile flags (edit the Armadillo path here) |
| `CXXFLAGS` | `-std=c++11 -g` | Additional C++ flags (debug symbols) |
| `LDFLAGS` | `-fopenmp` | Linker flags |

> **Tip**: For production runs, change `-O0` to `-O2` in `OPTS` for optimised builds.

---

## Running Simulations

TARANTULA reads all input from the current working directory. You must `cd` into a directory that contains an `inputs/` folder (and for TDSE_XFEL, a `matrix_elements/` folder).

```bash
cd examples/n2o_tdse
../../bin/tarantula
```

Output files are written to `outputs/` in the current working directory.

---

## Input File Reference

### `inputs/input.dat`

The main configuration file uses a keyword-value format (one per line, whitespace-delimited). Lines beginning with `#` are comments.

#### Common Keywords

| Keyword | Type | Description |
|---------|------|-------------|
| `CALC_TYPE` | string | `TDSE_XFEL` or `SPAWN_DECAY` |

#### TDSE_XFEL Keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `RWA` | bool | `false` | Rotating-wave approximation |
| `GAUSS` | bool | `true` | Gaussian pulse envelope (else constant field) |
| `DECAY` | bool | `true` | Include Auger/photoionisation decay widths |
| `FOCAL_AVG` | bool | `false` | Focal-volume averaging |
| `BANDW_AVG` | bool | `false` | Bandwidth averaging |
| `TWOPULSE` | bool | `false` | Two-pulse calculation |
| `PRINT_MAT` | bool | `false` | Print Hamiltonian matrix to stdout |
| `TWOSTATE` | bool | `false` | Force 2-state system |
| `STARK` | bool | `false` | Include AC-Stark shifts |
| `WRITE_PULSE` | bool | `false` | Write pulse shape to file |
| `DECAY_AMP` | bool | `false` | Track population loss channels |
| `ICALIB` | bool | `false` | Intensity calibration from saturation fluence |
| `DEBUG` | bool | `false` | Extra debug output |
| `TSTART` | double | `0.0` | Start time (fs); 0 → auto-set 6σ before pulse peak |
| `TEND` | double | — | End time (fs) |
| `DT` | double | — | Time step (fs) |
| `N_PRINT` | int | — | Print every N steps |
| `NEQN` | int | — | Number of electronic states |
| `ORIENT_AVG` | int | `1` | 1=none, 2=perpendicular avg (x,y), 3=full 3D avg (x,y,z) |
| `BW_SAMPLE_SIZE` | int | — | Number of bandwidth sampling points |
| `BW_EXTENT` | int | — | Bandwidth sampling extent (multiples of σ) |
| `FV_SAMPLE_SIZE` | int | — | Number of focal-volume intensity shells |
| `PULSE_1_FWHM` | double | — | Pulse 1 FWHM (fs) |
| `PULSE_1_TMAX` | double | — | Pulse 1 peak arrival time (fs) |
| `PULSE_1_WIDTH` | double | — | Pulse 1 bandwidth FWHM (eV) |
| `PULSE_1_SPOT_SIZE` | double | — | Pulse 1 focal spot size (μm²) |
| `PULSE_1_MU_X/Y/Z` | double | — | Pulse 1 polarisation vector components |
| `PULSE_2_*` | double | — | Same keywords for pulse 2 (when `TWOPULSE true`) |

#### SPAWN_DECAY Keywords

| Keyword | Type | Description |
|---------|------|-------------|
| `TSTART` | double | Start time (a.u. or fs depending on data) |
| `TEND` | double | End time |
| `DT` | double | Time step |
| `NEQN` | int | Total number of electronic states in the cascade |
| `NATOM` | int | Number of atoms in the molecule |
| `NSTEP` | int | Number of decay steps in the cascade |
| `NCHAN` | int | Number of decay channels per step |
| `NT_SPAWN` | int | Spawn a new TBF every NT_SPAWN time steps |

### Additional Input Files

#### TDSE_XFEL Mode

| File | Description |
|------|-------------|
| `inputs/photon_e_1.txt` | Central photon energies (eV), one per line |
| `inputs/intensity_1.txt` | Peak intensities (W/cm²), one per line |
| `inputs/auger_rates_1.txt` | Auger decay rates per state (eV) |
| `inputs/photoion_sigma_1.txt` | Photoionisation cross-sections per state (Mb) |
| `inputs/decay_channels.txt` | Decay channel types: `AUGER` or `PHOTO_TOTAL` |
| `inputs/icalib_states.txt` | State index pairs for intensity calibration |
| `matrix_elements/Diagonal.txt` | State energies (Hartree), one per line |
| `matrix_elements/Off_Diagonal_x.txt` | x-polarisation TDMs: `i j value` per line |
| `matrix_elements/Off_Diagonal_y.txt` | y-polarisation TDMs |
| `matrix_elements/Off_Diagonal_z.txt` | z-polarisation TDMs |

#### SPAWN_DECAY Mode

| File | Description |
|------|-------------|
| `inputs/data_files/data_J_I.txt` | AIMD trajectory data for channel J, step I. Per-timestep block: time, energy, KE, bondlength, then per-atom charges |
| `inputs/data_files/data_J_I_spin.txt` | Same format but with spin densities |
| `inputs/total_auger_rates.txt` | Total Auger rates per state (eV) |
| `inputs/partial_auger_rates_I.txt` | Partial Auger rates for decay step I (eV), one value per channel |

---

## Output Files

### TDSE_XFEL Mode

Output files are written to `outputs/`. Columns are whitespace-delimited.

| File Pattern | Content |
|--------------|---------|
| `<i>.txt` | Time-resolved state populations for calculation `i` |
| `<i>_x.txt`, `<i>_y.txt` | Populations for x/y polarisations (`ORIENT_AVG 2`) |
| `<i>_perp_avg.txt` | Perpendicular orientational average |
| `<i>_z.txt`, `<i>_oreint_avg.txt` | z-polarisation and full orientational average (`ORIENT_AVG 3`) |
| `energy_variable.txt` | Final-time populations vs photon energy |
| `intensity_variable.txt` | Final-time populations vs intensity |
| `pulse_<n>_<i>.txt` | Pulse electric field time profile |

**Column format**: `time  pop_0  pop_1  ...  pop_(N-1)  [decay_channels...]  norm`

### SPAWN_DECAY Mode

| File Pattern | Content |
|--------------|---------|
| `outputs/population_J.txt` | Time-resolved populations for state J (columns: time, TBF_0, TBF_1, ..., norm) |
| `outputs/bondlength_J.txt` | Bond lengths for each TBF of state J |
| `outputs/energy_J.txt` | Potential energies for each TBF |
| `outputs/kenergy_J.txt` | Kinetic energies for each TBF |
| `outputs/0_charge_J.txt` | Mulliken charges on atom 0 for each TBF |
| `outputs/1_charge_J.txt` | Mulliken charges on atom 1 for each TBF |
| `outputs/0_spin_J.txt` | Spin densities on atom 0 for each TBF |
| `outputs/1_spin_J.txt` | Spin densities on atom 1 for each TBF |

---

## Examples

### Example 1: N2O XFEL TDSE (`examples/n2o_tdse/`)

Reproduces the double core-hole XANES N2O wave packet simulation from:
> A. E. A. Fouda, D. Koulentianos, L. Young, G. Doumy and P. J. Ho, *Mol. Phys.* (2022) e2133749 
 
**Simulation details:**: 9 electronic states, Gaussian pulse (1.5 fs FWHM, 1.2 eV bandwidth), bandwidth averaging (100 points), perpendicular orientational averaging, Auger + photoionisation decay.

```bash
cd examples/n2o_tdse
../../bin/tarantula
# Output written to outputs/0_perp_avg.txt, outputs/0_x.txt, outputs/0_y.txt
```

### Example 2: IBr Auger-Meitner Spawning (`examples/ibr_spawn/`)

Reproduces IBr multi-step decay cascade dynamics from:
> A. E. A. Fouda, S. H. Southworth and P. J. Ho, *J. Chem. Theory Comput.* (2024) 20, 20, 8782-8794

**Simulation details**: 5 electronic states, 2 decay steps, 2 channels per step, TBF spawning every 5000 time steps, using AIMD trajectory data for potential energies and bond lengths.

```bash
cd examples/ibr_spawn
../../bin/tarantula
# Output: outputs/population_0.txt through population_4.txt, plus bond length/energy/charge/spin data
```

Python plotting scripts are provided in the example directory:
- `plotting_population_all.py` — overview of all state populations
- `plotting_spawn.py` — individual spawned TBF populations
- `plotting_spawn_weight_avg.py` — weight-averaged molecular properties

---

