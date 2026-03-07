# cubefx

<img src="cubefx_eq.png" width="400">

[Cliquez ici pour la version française](#cubefx-1)

Synthwave music is all about that retro-futuristic sound: punchy drums, shimmering synths, deep rumbling bass. Think 80s aesthetics meets modern production.
To give audio that neon-soaked character, we apply a **per-bin phase shift** across the frequency spectrum, an effect that colors the sound with a distinctive retro texture.

Your mission: make this pipeline blazingly fast on both CPU and CUDA, without sacrificing accuracy.

For this, you’ll use [CubeCL](https://github.com/tracel-ai/cubecl), a Rust-based framework for high-performance compute kernels that run on both CPU and GPU, letting you optimize parallelism and memory access from a single codebase.

## Audio Processing

A song is a continuous wave, also called a signal. To store it on a computer, we discretize it into samples. Each sample represents the amplitude of the signal at one point in time.

```
Continuous Wave:        ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿

Discretized Samples:    • • • • • • • • • • • •
                        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
                      Sample points taken at regular intervals
```

A typical sample rate is 44,100 Hz, meaning 44,100 samples per second. Audio can be mono, stereo, or more, meaning the signal has multiple channels running in parallel.

### The Fourier Transform

To work in the frequency domain, we apply a Fourier Transform.
This mathematical operation converts a time-domain signal into its frequency components, which is the set of sine waves at different frequencies whose sum recreates the original signal.

```
Time Domain (signal):     Frequency Domain (spectrum):

    ∿∿∿                   |     |              |
   ∿  ∿                   |     |     |        |
  ∿    ∿        ──→       |     |     |   |    |
 ∿      ∿                 └─────┴─────┴───┴────┴──
                          Low           Mid    High
 Complicated wave         Individual frequency components
```

Each frequency bin is represented as a complex number. In practice, we simply use a pair of floats (real and imaginary parts), or, for tensors, a pair of float tensors.
Since we start from real-valued audio samples, we use the **Real FFT (RFFT)**, which exploits symmetry to produce only the non-redundant half of the spectrum. The inverse (**IRFFT**) transforms back to samples.

### Windowing

We can't apply the FFT to the entire song at once because songs aren't periodic, and treating them as such loses information. Instead, we work on small overlapping windows. Neighboring windows overlap so that every part of the song is covered smoothly, avoiding clicks and artifacts at boundaries.

The windowing is **pre-applied** before your pipeline begins. You receive pre-windowed data and return processed windows. Signal reconstruction is out of scope today.

### The Phase Shift Effect

For each frequency bin `k` in the spectrum, we apply a rotation proportional to the bin index:

```
spectrum[k]  *=  e^(i·α·k)  =  cos(α·k) + i·sin(α·k)
```

This gives lower frequencies a small nudge and higher frequencies a larger rotation, creating a characteristic coloring of the signal. A single scalar `α` controls the strength of the effect.

### The Pipeline

Three kernels are executed: RFFT, Phase shift and IRFFT, and the benchmark measures the execution of the whole pipeline. When profiling, you may also see a PRNG (pseudo-random number generator) running to create data to work on, but this is not measured in the evaluated benchmarks.

```
Pre-windowed input   [num_windows, num_channels, window_length]
        │
        ▼
      RFFT           [num_windows, num_channels, freq_bins]   (complex: two float tensors)
        │
        ▼
  Phase shift        spectrum[k] *= e^(i·α·k)
        │
        ▼
      IRFFT          [num_windows, num_channels, window_length]
        │
        ▼
  Processed windows
```

For example, with `window_length := 2048`, the RFFT produces `freq_bins := window_length / 2 + 1 = 1025` frequency bins.

## Tensors, Shapes, and Memory Layout

A **tensor** is a multi-dimensional array. You can think of it as a generalization of a matrix to any number of dimensions. Throughout this codebase, audio data is stored as tensors. For example, the input to the pipeline has shape `[num_windows, num_channels, window_length]`: the first dimension indexes which window, the second which channel, and the third which sample within that window.

The **shape** tells you how many elements exist along each dimension. The **strides** tell you how to translate a multi-dimensional index into a flat memory offset. For a row-major tensor with shape `[4, 3]`, the strides are `[3, 1]`: stepping one index along dimension 0 skips 3 elements in memory, and stepping along dimension 1 skips 1.

```
Shape [4, 3], strides [3, 1]:

Logical layout:        Memory (flat):
┌───┬───┬───┐
│ 0 │ 1 │ 2 │          0  1  2  3  4  5  6  7  8  9 10 11
├───┼───┼───┤          ▲        ▲        ▲        ▲
│ 3 │ 4 │ 5 │          row 0    row 1    row 2    row 3
├───┼───┼───┤
│ 6 │ 7 │ 8 │
├───┼───┼───┤
│ 9 │10 │11 │
└───┴───┴───┘
```

**All tensors in this codebase are row-major**, meaning the last dimension is always contiguous in memory (stride 1). This matters for optimization: when threads access consecutive elements along the last dimension, memory loads can be coalesced on GPU and cache-friendly on CPU.

You won't need to compute memory offsets manually because `layout.rs` handles strides for you. If you're curious about the details, that's the place to look.

## Getting Started

### Running the Application

```bash
cargo run --release
```

Always use `--release` for reliable timings, otherwise it makes a debug build which is often much slower and not representative.

**Backend Selection:**

```bash
# CPU
cargo run --release --features cubecl/cpu

# CUDA
cargo run --release --features cubecl/cuda

# Default: WGPU (no feature flags needed)
```

You'll be evaluated on both CPU and CUDA, so benchmark with those configurations.

### Profiling

Enable the CubeCL profiler by setting `stdout` to `true` in `cubecl.toml` to identify bottleneck kernels.
You may also need to set the `CUBECL_DEBUG_OPTION` environment variable to `profile`:

```bash
export CUBECL_DEBUG_OPTION=profile
```

More configuration details can be found [here](https://burn.dev/books/cubecl/advanced-usage/config.html#configuration-file-structure:~:text=Configuration%20File%20Structure) although many settings will be overkill for this project (there are no streams nor autotuning).

### Testing Correctness

Run the test suite frequently:

```bash
cargo test
```

**Backend Selection:**

```bash
# CPU
cargo test --features cubecl/cpu

# CUDA
cargo test --features cubecl/cuda

# Default: WGPU
cargo test
```

**Critical Tests:**

- `large_fft_roundtrip_no_phase_shift`: Ensures FFT/IFFT roundtrip accuracy
- `small_fft_round_trip_with_phase_shift`: Verifies the complete effect pipeline

These two tests are the only ones that truly matter for evaluation. Tests inside `cubefx-engine` are there to help you debug along the way. You can look at the test and benchmark inputs to understand what assumptions you can reasonably make about the data (e.g. window size, number of channels).

### Debugging

**Test Mode:**

You can control how tests handle numerical and compilation errors via the `CUBE_TEST_MODE` environment variable:

- `Correct` (default): numerical errors fail, showing only the first error.
- `PrintFail`: print all tensor elements, showing which are wrong. Accepts an optional filter suffix to see parts of the data only.

```bash
export CUBE_TEST_MODE=Correct                  # default
export CUBE_TEST_MODE=PrintFail:.,10-20        # filter: all first dims, indices 10–20 on second
```

Filters are comma-separated dimension selectors: `.` for all indices, `M` for a single index, `M-N` for a range. The number of entries must match the tensor rank.
For more details, go [here](https://github.com/tracel-ai/cubek/blob/7b9a1f87d9e0cb984cfcb83fb0f04240513038e7/crates/cubek-test-utils/src/test_mode/base.rs).

**Generated Code:**

For CUDA or WGPU, you can print the generated code of each kernel by setting the `CUBECL_DEBUG_LOG` environment variable to `stdout`. More details [here](https://burn.dev/books/cubecl/advanced-usage/config.html#environment-variable-overrides:~:text=Environment%20Variable%20Overrides).
This might not print anything if you run a test that succeeds, so either make the test fail or add the `--nocapture` flag.

```bash
export CUBECL_DEBUG_LOG=stdout
cargo test --features cubecl/cuda -- --nocapture
```

Set the environment variable to `0` to deactivate it.

```bash
export CUBECL_DEBUG_LOG=0
```

## Code Structure

The project is split into two crates: **`cubefx-eval`** is the main binary, and **`cubefx-engine`** is the library where all your work lives.

### cubefx-eval (Do Not Modify)

Handles benchmarking and correctness testing. Your modifications here won't be used during evaluation since we'll use our own version.
Notice that the data type (`f32`) is selected in this crate, so choosing a smaller data type is not a possibily.

### cubefx-engine (Your Workspace)

All audio processing logic lives here. You can modify anything as long as:

1. The API used by `cubefx-eval` remains compatible
2. Both critical correctness tests pass

The backend is selected at compile time via CubeCL's `TestRuntime`, using `--features cubecl/cuda` or `--features cubecl/cpu`.

**Files:**

- **`base.rs`**: Do not modify. Contains the entry point called by `cubefx-eval`.
- **`cube/`**
  - **`phase_shift.rs`**: CubeCL kernel and launch code for per-bin phase shifting
  - **`layout.rs`**: Converts tensors to views over a single batch element at a time, handling batch offsets and strides
  - **`tests/`**: RFFT and IRFFT tests, verified against a pure Rust reference implementation
  - **`fft/`**
    - **`rfft.rs`**: CubeCL kernel and launch code for the Real FFT
    - **`irfft.rs`**: CubeCL kernel and launch code for the inverse RFFT
    - **`fft_inner.rs`**: Shared inner compute for both FFT directions. There are no global memory I/O or dispatch here, just the core arithmetic. More complex to understand and likely harder to optimize; approach with care.

Most of your modifications will be in `phase_shift.rs`, `rfft.rs`, and `irfft.rs`. You may also find opportunities in `layout.rs` and `fft_inner.rs`, although it's probably harder.

## Optimization Opportunities

Normally, the phase shift kernel should execute much faster than the other two, even in their current suboptimal form.
If it's not the case, the phase shift kernel is probably doing something stupid.
Once this is fixed, you can move on to less trivial kernel modifications.

### Launch Configuration

The default kernels are extremely naive: a single worker processes the entire input sequentially.

CubeCL kernels are launched with a _cube count_ and a _cube dim_. At the moment, all _cube dims_ and _cube counts_ are hardcoded to 1.

- **_cube count_**: The number of independent tasks that can run on different streaming processors.  
  On CUDA, this maps to blocks. On CPU, this creates a loop over all tasks.  
  Inside the kernel, you can access the current cube ID via `CUBE_POS` (or `CUBE_POS_X`, `CUBE_POS_Y`, `CUBE_POS_Z` if using multiple dimensions).

- **_cube dim_**: The number of workers per cube (called _units_ in CubeCL).  
  On CUDA, this maps to threads. On CPU, this also maps to threads (cores).

  Within a cube, units are grouped into _planes_ (called warps on CUDA).  
  On CPU, the plane size is 1 (one unit = one plane).

  Units within a plane execute in lockstep: they access memory at the same time and follow the same code path (unless there is divergence).

  You can query the plane size at runtime with:
  `client.properties().hardware.plane_size_max`

  It is good practice to define `cube_dim` as 2D:  
  `(plane_size, number_of_planes)`

  Inside the kernel:
  - `CUBE_DIM_X`: plane size
  - `CUBE_DIM_Y`: number of planes
  - `UNIT_POS_X`: unit ID within the plane
  - `UNIT_POS_Y`: plane ID within the cube

### Memory Access Patterns

How your threads access memory has a major impact on performance:

- **GPU:** Threads within the same plane (warp) should access consecutive memory addresses. This is called _memory coalescing_.  
  For example, if unit 0 reads address 0, unit 1 reads address 1, … up to unit 31, the GPU can fetch all in a single transaction.  
  Strided or scattered access reduces bandwidth efficiency.

- **CPU:** Each thread should work on a contiguous block of memory that fits in the CPU cache.  
  Strided or scattered access forces threads to load data from different cache lines, increasing cache misses and slowing down execution.  
  CPUs prefer sequential access per thread to maximize cache line utilization and prefetching.

These requirements can sometimes conflict, so it is recommended to query hardware properties (e.g., plane size) and design your kernel to handle both backends efficiently.

### Vectorization

CubeCL supports a "line" abstraction that lets each thread process multiple elements per memory transaction. None of the kernels currently use this. Enabling it means fewer global memory accesses for the same amount of work, which compounds well with good access patterns.
In hardware properties, there is a `load_width` specifying how many bits can be loaded at the same time by one unit.
The maximal vectorization factor for your backend is `load_width` divided by the number of bits of each element (how many bits are there in an `f32`?).
A good vectorization factor can drastically speedup global memory reads and writes, especially on GPU. On CPU, it's more important at speeding the compute, but the FFT inner compute might be very hard to vectorize.

To be able to vectorize a tensor, the dimension in which elements are contiguous in memory (the last dimension in our case because everything is in row-major order),
must be divisible by the vectorization factor.
While this should be the case for windows of signal samples which are assumed to be a power of 2 (typically, `window_length=2048 elements`),
this won't be the case for the spectrums, because of the formula `freq_bins = window_length / 2 + 1 = 1025`. Perhaps padding each window of frequency bins (to have a shape of say, 1032) with zeros would help.

### Kernel Fusion

Kernel fusion combines multiple operations into a single kernel, allowing intermediate results to stay in registers or shared memory rather than being written back and forth to global memory.

Currently, the FFT, phase shift, and IRFFT are separate kernel launches.

In principle, these three kernels could be merged into a single kernel (or at least two), since they have similar launch configurations, and data in the FFT is already in shared memory before being written to global memory.

### Others

- **Unrolling**: CubeCL supports loop unrolling via `#[unroll]` over `for` loops, but only when the loop range is a **comptime** value. Comptime values are a CubeCL concept: constants baked into the kernel at compile time rather than passed as runtime arguments. If a loop bound is a runtime variable, it cannot be unrolled. When applicable, unrolling eliminates branch overhead and can expose more instruction-level parallelism to the compiler.

- **Replace while loops with for loops**: while loops can create unpredictable branching within a plane, leading to divergence and reduced performance. Using for loops with known bounds makes execution more predictable.

- **Balance workloads**: Ideally, on GPU, the _cube count_ should spread equally across streaming multiprocessors (SMs). The number of SMs is available in the hardware properties.

- **Respect the hardware**: Using too many planes at once may exceed what the system can efficiently support. The same applies to registers, shared memory, and cache. These resources are limited, and pushing them too far can silently fallback to less efficient behaviors. In high-performance, everything is a tradeoff.

## Evaluation and Ranking

### Disqualification Criteria

Teams that produce incorrect results on either CPU or CUDA will be disqualified and excluded from ranking statistics.

### Scoring System

All teams will be evaluated on the same machine for each backend. Your score is calculated as:

$$
\text{score} = \frac{\text{mean}_{\text{CPU}} - \text{duration}_{\text{CPU}}}{\text{std}_{\text{CPU}}} + \frac{\text{mean}_{\text{CUDA}} - \text{duration}_{\text{CUDA}}}{\text{std}_{\text{CUDA}}}
$$

Where `mean` and `std` are computed across all qualifying teams for each backend, and `duration` is your team's benchmark time (mean of 10 runs after warmup). **Higher is better.**

### Tiebreaker

Due to the nondeterministic nature of benchmarks, scores that are extremely close may be considered tied.

If this occurs, the tied teams’ submissions will be benchmarked again on a different machine using additional backends (AMD and/or Metal). These results will be used to determine the final ranking.

### Submission

First, fork this repository (use the `Fork` button in the upper-right corner of the GitHub page).
During the competition, you will be asked to provide the link to your fork.

Only your **final commit** will be evaluated. Make sure it:

1. Passes both critical correctness tests
2. Includes your optimization work
3. Does not introduce new dependencies beyond what's provided

If you discover an issue with your final commit after submission, it's worth reaching out, but no guarantees can be made.

## Tips

1. **Profile first.** Use the CubeCL profiler (set stdout to true in `cubecl.toml`) to identify bottlenecks before changing anything. The codebase is small enough that you can also reason about where time is spent, but measurements are more reliable.

2. **I/O dominates compute, especially on GPU.** Reading from and writing to global memory is far more expensive than the arithmetic itself. On GPU in particular, the biggest gains usually come from reducing how much data you move, not from squeezing more flops out of the compute. On CPU the gap is smaller, but memory access patterns still matter a lot.

3. **Verify correctness.** Run `cargo test` after every change. Incorrect results will get you disqualified! Slower but correct is always better.

4. **Commit often.** Each time you have better performance and passing tests, save that state. It's easy to break things; checkpoints let you recover.

5. **Start small.** Make incremental changes and test frequently. It's easier to debug one change than a massive rewrite.

6. **Balance backends.** Don't over-optimize for one at the expense of the other. Your score depends on both CPU and CUDA.

7. **Accept temporary breakage.** Removing code to measure how much it costs is a valid way to build intuition. You'll lose correctness for a moment but gain insight on performance impact of some lines of code. Just don't submit in that state.

8. **Verify if a change is worth pursuing.** Following point 7, before optimizing code, check if removing the code altogether actually speeds up anything.

---

# cubefx

<img src="cubefx_eq.png" width="400">

La musique _synthwave_ est avant tout une question de son rétro-futuriste : batteries percutantes, synthés scintillants, basse grave et profonde. Pensez à l'esthétique des années 80 rencontre la production moderne.
Pour donner à l'audio ce caractère aux néons, on applique un **décalage de phase par bin** (_per-bin phase shift_) sur tout le spectre de fréquences — un effet qui colore le son avec une texture rétro caractéristique.

Votre mission : rendre ce _pipeline_ incroyablement rapide sur _CPU_ et _CUDA_, sans sacrifier la précision.

Pour cela, vous utiliserez [CubeCL](https://github.com/tracel-ai/cubecl), un _framework_ Rust pour des _kernels_ de calcul haute performance qui s'exécutent à la fois sur _CPU_ et _GPU_, vous permettant d'optimiser le parallélisme et les accès mémoire depuis une seule base de code.

## Traitement audio

Une chanson est une onde continue, aussi appelée signal. Pour la stocker sur un ordinateur, on la discrétise en échantillons (_samples_). Chaque échantillon représente l'amplitude du signal à un instant donné.

```
Onde continue :         ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿

Échantillons discrets : • • • • • • • • • • • •
                        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
                      Points d'échantillonnage à intervalles réguliers
```

Un taux d'échantillonnage typique est de 44 100 Hz, soit 44 100 échantillons par seconde. L'audio peut être mono, stéréo ou plus, ce qui signifie que le signal comporte plusieurs canaux en parallèle.

### La transformée de Fourier

Pour travailler dans le domaine fréquentiel, on applique une transformée de Fourier.
Cette opération mathématique convertit un signal dans le domaine temporel en ses composantes fréquentielles, c'est-à-dire l'ensemble des ondes sinusoïdales à différentes fréquences dont la somme recrée le signal original.

```
Domaine temporel (signal) :   Domaine fréquentiel (spectre) :

    ∿∿∿                       |     |              |
   ∿  ∿                       |     |     |        |
  ∿    ∿        ──→           |     |     |   |    |
 ∿      ∿                     └─────┴─────┴───┴────┴──
                              Bas         Milieu   Haut
 Onde complexe                Composantes fréquentielles individuelles
```

Chaque bin de fréquence est représenté par un nombre complexe. En pratique, on utilise simplement une paire de nombres flottants (parties réelle et imaginaire), ou, pour les tenseurs, une paire de tenseurs de nombres flottants.
Puisqu'on part d'échantillons audio à valeurs réelles, on utilise la **_Real FFT_ (_RFFT_)**, qui exploite la symétrie pour ne produire que la moitié non redondante du spectre. L'inverse (**IRFFT**) retransforme en échantillons.

### Le fenêtrage (_Windowing_)

On ne peut pas appliquer la _FFT_ à toute la chanson d'un coup, car les chansons ne sont pas périodiques, et les traiter ainsi entraîne une perte d'information. On travaille donc sur de petites fenêtres qui se chevauchent. Les fenêtres voisines se superposent pour que chaque partie de la chanson soit couverte en douceur, évitant les clics et artefacts aux frontières.

Le fenêtrage est **pré-appliqué** avant le début de votre _pipeline_. Vous recevez des données déjà fenêtrées et retournez des fenêtres traitées. La reconstruction du signal est hors sujet aujourd'hui.

### L'effet de décalage de phase

Pour chaque bin de fréquence `k` dans le spectre, on applique une rotation proportionnelle à l'indice du bin :

```
spectrum[k]  *=  e^(i·α·k)  =  cos(α·k) + i·sin(α·k)
```

Cela donne aux basses fréquences une légère nudge et aux hautes fréquences une rotation plus importante, créant une coloration caractéristique du signal. Un scalaire unique `α` contrôle l'intensité de l'effet.

### Le pipeline

Trois _kernels_ sont exécutés : _RFFT_, décalage de phase et _IRFFT_. Le _benchmark_ mesure l'exécution de l'ensemble du _pipeline_. Lors du profilage, vous pourrez aussi voir un _PRNG_ (_pseudo-random number generator_) tournant pour créer des données, mais celui-ci n'est pas mesuré dans les _benchmarks_ évalués.

```
Entrée pré-fenêtrée   [num_windows, num_channels, window_length]
        │
        ▼
      RFFT           [num_windows, num_channels, freq_bins]   (complexe : deux tenseurs de flottants)
        │
        ▼
  Décalage de phase  spectrum[k] *= e^(i·α·k)
        │
        ▼
      IRFFT          [num_windows, num_channels, window_length]
        │
        ▼
  Fenêtres traitées
```

Par exemple, avec `window_length := 2048`, la _RFFT_ produit `freq_bins := window_length / 2 + 1 = 1025` bins de fréquence.

## Tenseurs, formes et disposition mémoire

Un **tenseur** est un tableau multidimensionnel. On peut le voir comme une généralisation d'une matrice à un nombre quelconque de dimensions. Dans cette base de code, les données audio sont stockées sous forme de tenseurs. Par exemple, l'entrée du _pipeline_ a la forme `[num_windows, num_channels, window_length]` : la première dimension indexe la fenêtre, la seconde le canal, et la troisième l'échantillon dans la fenêtre.

La **forme** (_shape_) indique combien d'éléments existent le long de chaque dimension. Les **strides** (_strides_) indiquent comment traduire un indice multidimensionnel en décalage mémoire plat. Pour un tenseur _row-major_ de forme `[4, 3]`, les _strides_ sont `[3, 1]` : avancer d'un indice sur la dimension 0 saute 3 éléments en mémoire, et avancer sur la dimension 1 en saute 1.

```
Forme [4, 3], strides [3, 1] :

Disposition logique :  Mémoire (plate) :
┌───┬───┬───┐
│ 0 │ 1 │ 2 │          0  1  2  3  4  5  6  7  8  9 10 11
├───┼───┼───┤          ▲        ▲        ▲        ▲
│ 3 │ 4 │ 5 │          ligne 0  ligne 1  ligne 2  ligne 3
├───┼───┼───┤
│ 6 │ 7 │ 8 │
├───┼───┼───┤
│ 9 │10 │11 │
└───┴───┴───┘
```

**Tous les tenseurs de cette base de code sont _row-major_**, ce qui signifie que la dernière dimension est toujours contiguë en mémoire (_stride_ 1). Cela importe pour l'optimisation : lorsque les _threads_ accèdent à des éléments consécutifs sur la dernière dimension, les chargements mémoire peuvent être coalescés sur _GPU_ et favorables au cache sur _CPU_.

Vous n'aurez pas besoin de calculer manuellement les décalages mémoire, car `layout.rs` gère les _strides_ à votre place. Si vous êtes curieux des détails, c'est l'endroit où regarder.

## Démarrage

### Lancer l'application

```bash
cargo run --release
```

Utilisez toujours `--release` pour des mesures fiables ; sinon, une version de débogage est compilée, souvent bien plus lente et non représentative.

**Sélection du backend :**

```bash
# CPU
cargo run --release --features cubecl/cpu

# CUDA
cargo run --release --features cubecl/cuda

# Défaut : WGPU (aucun flag nécessaire)
```

Vous serez évalué sur _CPU_ et _CUDA_, donc faites vos _benchmarks_ avec ces configurations.

### Profilage

Activez le profileur CubeCL en mettant `stdout` à `true` dans `cubecl.toml` pour identifier les _kernels_ goulots d'étranglement.
Vous devrez peut-être aussi définir la variable d'environnement `CUBECL_DEBUG_OPTION` à `profile` :

```bash
export CUBECL_DEBUG_OPTION=profile
```

Plus de détails de configuration sont disponibles [ici](https://burn.dev/books/cubecl/advanced-usage/config.html#configuration-file-structure) bien que de nombreux paramètres soient superflus pour ce projet (pas de _streams_ ni d'_autotuning_).

### Tester la correction

Lancez la suite de tests fréquemment :

```bash
cargo test
```

**Sélection du backend :**

```bash
# CPU
cargo test --features cubecl/cpu

# CUDA
cargo test --features cubecl/cuda

# Défaut : WGPU
cargo test
```

**Tests critiques :**

- `large_fft_roundtrip_no_phase_shift` : vérifie la précision de l'aller-retour _FFT_/_IFFT_
- `small_fft_round_trip_with_phase_shift` : valide le _pipeline_ d'effet complet

Ces deux tests sont les seuls qui comptent vraiment pour l'évaluation. Les tests dans `cubefx-engine` sont là pour vous aider à déboguer en cours de route. Consultez les entrées des tests et _benchmarks_ pour comprendre les hypothèses raisonnables sur les données (ex. taille de fenêtre, nombre de canaux).

### Débogage

**Mode test :**

Vous pouvez contrôler la façon dont les tests gèrent les erreurs numériques et de compilation via la variable d'environnement `CUBE_TEST_MODE` :

- `Correct` (défaut) : les erreurs numériques font échouer le test, en affichant uniquement la première erreur.
- `PrintFail` : affiche tous les éléments du tenseur, en indiquant lesquels sont erronés. Accepte un suffixe de filtre optionnel pour ne voir qu'une partie des données.

```bash
export CUBE_TEST_MODE=Correct                  # défaut
export CUBE_TEST_MODE=PrintFail:.,10-20        # filtre : toutes les premières dims, indices 10–20 sur la seconde
```

Les filtres sont des sélecteurs de dimension séparés par des virgules : `.` pour tous les indices, `M` pour un indice unique, `M-N` pour une plage. Le nombre d'entrées doit correspondre au rang du tenseur.
Pour plus de détails, consultez [ici](https://github.com/tracel-ai/cubek/blob/7b9a1f87d9e0cb984cfcb83fb0f04240513038e7/crates/cubek-test-utils/src/test_mode/base.rs).

**Code généré :**

Pour _CUDA_ ou _WGPU_, vous pouvez afficher le code généré de chaque _kernel_ en définissant la variable d'environnement `CUBECL_DEBUG_LOG` à `stdout`. Plus de détails [ici](https://burn.dev/books/cubecl/advanced-usage/config.html#environment-variable-overrides).
Cela peut ne rien afficher si le test réussit ; faites échouer le test ou ajoutez le flag `--nocapture`.

```bash
export CUBECL_DEBUG_LOG=stdout
cargo test --features cubecl/cuda -- --nocapture
```

Mettez la variable à `0` pour la désactiver.

```bash
export CUBECL_DEBUG_LOG=0
```

## Structure du code

Le projet est divisé en deux _crates_ : **`cubefx-eval`** est le binaire principal, et **`cubefx-engine`** est la bibliothèque où réside tout votre travail.

### cubefx-eval (Ne pas modifier)

Gère le _benchmarking_ et les tests de correction. Vos modifications ici ne seront pas utilisées lors de l'évaluation, car nous utiliserons notre propre version.
Notez que le type de données (`f32`) est sélectionné dans cette _crate_, donc choisir un type plus petit n'est pas une option.

### cubefx-engine (Votre espace de travail)

Toute la logique de traitement audio réside ici. Vous pouvez tout modifier, à condition que :

1. L'API utilisée par `cubefx-eval` reste compatible
2. Les deux tests de correction critiques passent

Le _backend_ est sélectionné à la compilation via `TestRuntime` de CubeCL, en utilisant `--features cubecl/cuda` ou `--features cubecl/cpu`.

**Fichiers :**

- **`base.rs`** : Ne pas modifier. Contient le point d'entrée appelé par `cubefx-eval`.
- **`cube/`**
  - **`phase_shift.rs`** : _Kernel_ CubeCL et code de lancement pour le décalage de phase par bin
  - **`layout.rs`** : Convertit les tenseurs en vues sur un seul élément de _batch_ à la fois, gérant les décalages de _batch_ et les _strides_
  - **`tests/`** : Tests _RFFT_ et _IRFFT_, vérifiés contre une implémentation de référence en Rust pur
  - **`fft/`**
    - **`rfft.rs`** : _Kernel_ CubeCL et code de lancement pour la _Real FFT_
    - **`irfft.rs`** : _Kernel_ CubeCL et code de lancement pour l'_IRFFT_
    - **`fft_inner.rs`** : Calcul interne partagé pour les deux directions _FFT_. Pas d'E/S mémoire globale ni de _dispatch_ ici, seulement l'arithmétique de base. Plus complexe à comprendre et probablement plus difficile à optimiser ; à aborder avec précaution.

La plupart de vos modifications se feront dans `phase_shift.rs`, `rfft.rs` et `irfft.rs`. Vous trouverez peut-être aussi des opportunités dans `layout.rs` et `fft_inner.rs`, bien que ce soit probablement plus difficile.

## Opportunités d'optimisation

Normalement, le _kernel_ de décalage de phase devrait s'exécuter beaucoup plus rapidement que les deux autres, même dans leur forme sous-optimale actuelle.
Si ce n'est pas le cas, le _kernel_ de décalage de phase fait probablement quelque chose de stupide.
Une fois cela corrigé, vous pouvez passer à des modifications de _kernels_ moins triviales.

### Configuration de lancement

Les _kernels_ par défaut sont extrêmement naïfs : un seul _worker_ traite toute l'entrée séquentiellement.

Les _kernels_ CubeCL sont lancés avec un _cube count_ et un _cube dim_. Pour l'instant, tous les _cube dims_ et _cube counts_ sont codés en dur à 1.

- **_cube count_** : Le nombre de tâches indépendantes pouvant s'exécuter sur différents processeurs de flux.  
  Sur _CUDA_, cela correspond aux _blocks_. Sur _CPU_, cela crée une boucle sur toutes les tâches.  
  Dans le _kernel_, vous pouvez accéder à l'identifiant du _cube_ courant via `CUBE_POS` (ou `CUBE_POS_X`, `CUBE_POS_Y`, `CUBE_POS_Z` si on utilise plusieurs dimensions).

- **_cube dim_** : Le nombre de _workers_ par _cube_ (appelés _units_ dans CubeCL).  
  Sur _CUDA_, cela correspond aux _threads_. Sur _CPU_, cela correspond aussi aux _threads_ (cœurs).

  Dans un _cube_, les _units_ sont regroupés en _planes_ (appelés _warps_ sur _CUDA_).  
  Sur _CPU_, la taille de _plane_ est 1 (une _unit_ = un _plane_).

  Les _units_ dans un _plane_ s'exécutent en _lockstep_ : ils accèdent à la mémoire en même temps et suivent le même chemin de code (sauf en cas de divergence).

  Vous pouvez interroger la taille de _plane_ à l'exécution avec :
  `client.properties().hardware.plane_size_max`

  Il est recommandé de définir `cube_dim` en 2D :  
  `(plane_size, number_of_planes)`

  Dans le _kernel_ :
  - `CUBE_DIM_X` : taille du _plane_
  - `CUBE_DIM_Y` : nombre de _planes_
  - `UNIT_POS_X` : identifiant de l'_unit_ dans le _plane_
  - `UNIT_POS_Y` : identifiant du _plane_ dans le _cube_

### Motifs d'accès mémoire

La façon dont vos _threads_ accèdent à la mémoire a un impact majeur sur les performances :

- **_GPU_ :** Les _threads_ dans le même _plane_ (_warp_) devraient accéder à des adresses mémoire consécutives. C'est ce qu'on appelle la _coalescence mémoire_ (_memory coalescing_).  
  Par exemple, si l'_unit_ 0 lit l'adresse 0, l'_unit_ 1 lit l'adresse 1, … jusqu'à l'_unit_ 31, le _GPU_ peut tout charger en une seule transaction.  
  Un accès strié ou dispersé réduit l'efficacité de la bande passante.

- **_CPU_ :** Chaque _thread_ devrait travailler sur un bloc continu de mémoire qui tient dans le cache _CPU_.  
  Un accès strié ou dispersé force les _threads_ à charger des données depuis différentes lignes de cache, augmentant les _cache misses_ et ralentissant l'exécution.  
  Les _CPU_ préfèrent un accès séquentiel par _thread_ pour maximiser l'utilisation des lignes de cache et le _prefetching_.

Ces exigences peuvent parfois être contradictoires, il est donc recommandé d'interroger les propriétés matérielles (ex. taille de _plane_) et de concevoir votre _kernel_ pour gérer efficacement les deux _backends_.

### Vectorisation

CubeCL supporte une abstraction « _line_ » qui permet à chaque _thread_ de traiter plusieurs éléments par transaction mémoire. Aucun _kernel_ ne l'utilise actuellement. L'activer signifie moins d'accès à la mémoire globale pour la même quantité de travail, ce qui se combine bien avec de bons motifs d'accès.
Dans les propriétés matérielles, il y a un `load_width` spécifiant combien de bits peuvent être chargés en même temps par une _unit_.
Le facteur de vectorisation maximal pour votre _backend_ est `load_width` divisé par le nombre de bits de chaque élément (combien de bits y a-t-il dans un `f32` ?).
Un bon facteur de vectorisation peut drastiquement accélérer les lectures et écritures en mémoire globale, surtout sur _GPU_. Sur _CPU_, c'est plus utile pour accélérer le calcul, mais le calcul interne de la _FFT_ peut être très difficile à vectoriser.

Pour pouvoir vectoriser un tenseur, la dimension dans laquelle les éléments sont contigus en mémoire (la dernière dimension dans notre cas, car tout est en ordre _row-major_) doit être divisible par le facteur de vectorisation.
Bien que ce soit le cas pour les fenêtres d'échantillons de signal qui sont supposées être une puissance de 2 (typiquement `window_length=2048 éléments`), ce ne sera pas le cas pour les spectres, à cause de la formule `freq_bins = window_length / 2 + 1 = 1025`. Peut-être que _padder_ chaque fenêtre de bins de fréquence (pour avoir une forme de, disons, 1032) avec des zéros pourrait aider.

### Fusion de kernels (_Kernel Fusion_)

La fusion de _kernels_ combine plusieurs opérations en un seul _kernel_, permettant aux résultats intermédiaires de rester dans les registres ou la mémoire partagée plutôt que d'être écrits et relus depuis la mémoire globale.

Actuellement, la _FFT_, le décalage de phase et l'_IRFFT_ sont des lancements de _kernels_ séparés.

En principe, ces trois _kernels_ pourraient être fusionnés en un seul (ou au moins deux), puisqu'ils ont des configurations de lancement similaires, et les données de la _FFT_ sont déjà en mémoire partagée avant d'être écrites en mémoire globale.

### Autres

- **_Unrolling_** : CubeCL supporte le déroulage de boucles via `#[unroll]` sur les boucles `for`, mais uniquement lorsque la plage de la boucle est une valeur **comptime**. Les valeurs _comptime_ sont un concept CubeCL : des constantes intégrées dans le _kernel_ à la compilation plutôt que passées en arguments à l'exécution. Si la borne d'une boucle est une variable à l'exécution, elle ne peut pas être déroulée. Quand applicable, le déroulage élimine le surcoût des branchements et peut exposer davantage de parallélisme au niveau instruction au compilateur.

- **Remplacer les boucles `while` par des boucles `for`** : les boucles `while` peuvent créer des branchements imprévisibles dans un _plane_, entraînant divergence et performances réduites. Utiliser des boucles `for` avec des bornes connues rend l'exécution plus prévisible.

- **Équilibrer les charges de travail** : Idéalement, sur _GPU_, le _cube count_ devrait se répartir équitablement entre les processeurs de flux (_SMs_ — _streaming multiprocessors_). Le nombre de _SMs_ est disponible dans les propriétés matérielles.

- **Respecter le matériel** : Utiliser trop de _planes_ à la fois peut dépasser ce que le système peut supporter efficacement. Il en va de même pour les registres, la mémoire partagée et le cache. Ces ressources sont limitées, et les pousser trop loin peut silencieusement basculer vers des comportements moins efficaces. En haute performance, tout est un compromis.

## Évaluation et classement

### Critères de disqualification

Les équipes produisant des résultats incorrects sur _CPU_ ou _CUDA_ seront disqualifiées et exclues des statistiques de classement.

### Système de notation

Toutes les équipes seront évaluées sur la même machine pour chaque _backend_. Votre score est calculé comme suit :

$$
\text{score} = \frac{\text{mean}_{\text{CPU}} - \text{duration}_{\text{CPU}}}{\text{std}_{\text{CPU}}} + \frac{\text{mean}_{\text{CUDA}} - \text{duration}_{\text{CUDA}}}{\text{std}_{\text{CUDA}}}
$$

Où `mean` et `std` sont calculés sur toutes les équipes qualifiées pour chaque _backend_, et `duration` est le temps de _benchmark_ de votre équipe (moyenne de 10 exécutions après préchauffage). **Plus c'est élevé, mieux c'est.**

### Départage

En raison de la nature non déterministe des _benchmarks_, des scores extrêmement proches peuvent être considérés comme ex æquo.

Si cela se produit, les soumissions des équipes à égalité seront de nouveau _benchmarkées_ sur une machine différente en utilisant des _backends_ supplémentaires (_AMD_ et/ou _Metal_). Ces résultats seront utilisés pour déterminer le classement final.

### Soumission

Commencez par forker ce dépôt (utilisez le bouton `Fork` dans le coin supérieur droit de la page GitHub).
Durant la compétition, on vous demandera de fournir le lien vers votre fork.

Seul votre **dernier _commit_** sera évalué. Assurez-vous qu'il :

1. Passe les deux tests de correction critiques
2. Inclut votre travail d'optimisation
3. N'introduit pas de nouvelles dépendances au-delà de ce qui est fourni

Si vous découvrez un problème avec votre dernier _commit_ après la soumission, cela vaut la peine de nous contacter, mais aucune garantie ne peut être donnée.

## Conseils

1. **Profilez d'abord.** Utilisez le profileur CubeCL (mettez `stdout` à `true` dans `cubecl.toml`) pour identifier les goulots d'étranglement avant de changer quoi que ce soit. La base de code est assez petite pour raisonner sur les temps d'exécution, mais les mesures sont plus fiables.

2. **Les Entrées/Sorties dominent le calcul, surtout sur _GPU_.** Lire et écrire en mémoire globale est bien plus coûteux que l'arithmétique elle-même. Sur _GPU_ en particulier, les plus grands gains viennent généralement de la réduction du volume de données déplacées, pas d'extraire plus de _flops_ du calcul.

3. **Vérifiez la correction.** Lancez `cargo test` après chaque modification. Des résultats incorrects vous disqualifieront ! Plus lent mais correct vaut toujours mieux.

4. **Commitez souvent.** Chaque fois que vous avez de meilleures performances et des tests qui passent, sauvegardez cet état. Il est facile de casser quelque chose ; les points de sauvegarde permettent de récupérer.

5. **Commencez petit.** Faites des changements incrémentaux et testez fréquemment. Il est plus facile de déboguer un changement qu'une réécriture massive.

6. **Équilibrez les _backends_.** Ne sur-optimisez pas pour l'un au détriment de l'autre. Votre score dépend des deux, _CPU_ et _CUDA_.

7. **Acceptez la casse temporaire.** Supprimer du code pour mesurer combien il coûte est une façon valide de construire une intuition. Vous perdrez l'aspect correct pendant un moment mais gagnerez en compréhension de l'impact sur les performances de certaines lignes de code. Ne soumettez juste pas dans cet état.

8. **Vérifiez si un changement vaut la peine d'être poursuivi.** Suite au point 7, avant d'optimiser du code, vérifiez que supprimer entièrement ce code accélère effectivement quelque chose.
