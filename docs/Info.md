# Behavioral Data — Overview

## The Animals

Data comes from **3 mice** which are about 2 months old: M06, M09, and M19— the dataset however comprises in total 5 animals. 

Only M09 has pre-processed CSV files — M06 and M19 only have the raw H5 files. The animals were recorded across many sessions spanning several weeks, with weekend breaks separating blocks of consecutive training days.

---

## The Task

This is a **Go/Nogo visual angle discrimination task**. The mouse is presented with a visual stimulus on a screen for 2s. There are two possible stimuli—corresponding to the different orientations of the patterns on the sliding screen :

- `stim = +1` (e.g., vertical pattern) → the "Go" stimulus. The correct response is to **lick**.
- `stim = −1` (e.g., horizontal pattern) → the "Nogo" stimulus. The correct response is to **withhold licking**.

After the stimulus is shown, there is an **answer window** of 3s—a time period during which the animal's lick (`licks = +1`)  or silence (`licks = 0`) is recorded as its response. This defines four possible conditions: 
- `state = 1` (« Hit ») when `stim = +1` and `licks = +1`
- `state = 2` (« Miss ») when `stim = +1` and `licks = 0`
- `state = 3` (« False alarm ») when `stim = -1` and `licks = +1`
- `state = 4` (« Correct reject ») when `stim = -1` and `licks = 0`

### Reward
The mouse receives feedback: the reward (water) is only delivered on correct go trials (hits). A false alarm signal (a loud sound) is delivered only when the animal licks when it is not supposed to (false alarm). Therefore,
- `reward = +1` if `state = 1` (« Hit »)
- `reward = 0` if `state = 2` (« Miss ») or if `state = 4` (« Correct reject »)
- `reward = -1` if `state = 3` (« False alarm »)

### Answer window 
If `answer_window = 0`, the mouse licked outside the 3s answer window (e.g., the animal licked during the inter-trial interval, aborting the trial early).

---

## The Raw H5 Variables

Each H5 file contains **one value per trial** for four binary variables:

| Variable | Values | Description |
|----------|-------|-|
| `stim` | +1 / −1 | Which stimulus was shown (Go / Nogo) |
| `licks` | 0 / 1 | Whether the animal licked during the answer window |
| `reward` | -1 / 0 / 1 | Whether a reward was delivered (redundant with stim+lick, useful as sanity check) |
| `answer_window` | 0 / 1 | Whether the trial had a valid response window |

The files also contain two **index arrays**:

- **`sessions`** — trial indices marking the **end of each session**. For example, if `sessions[0] = 150`, the first session spans trials 0–149.
- **`weekends`** — trial indices where a **multi-day break** occurred, separating the recording period into blocks of consecutive training days.

---

## Dataset Sizes

| Animal | Total trials | Sessions | Weekend break indices |
|--------|-------------|----------|-----------------------|
| M06 | 8,396 | 66 | 1499, 4528 |
| M09 | 5,684 | 44 | 1350, 3752 |

M06 has notably more trials and sessions than M09, suggesting it was trained for longer or had more trials per session on average.

---

## The 5 Behavioral States (M09 CSVs)

The collaborators pre-computed a **state label for every trial** by combining the three raw variables. Each trial falls into exactly one of five categories:

| State | Name | Condition | Meaning | % of trials |
|-------|------|-----------|---------|-------------|
| 1 | **Hit** | stim=+1, lick=1 | Go stimulus shown, animal licked correctly → rewarded | ~33% |
| 2 | **Miss** | stim=+1, lick=0 | Go stimulus shown, animal failed to lick | ~12% |
| 3 | **False Alarm** | stim=−1, lick=1 | Nogo stimulus shown, animal licked incorrectly | ~15% |
| 4 | **Correct Reject** | stim=−1, lick=0 | Nogo stimulus shown, animal correctly withheld | ~21% |
| 5 | **No window** | answer_window=0 | No valid response window (aborted/invalid trial) | ~19% |

---

## What the Data Tells Us About Behavior

The state distribution alone reveals clear behavioral variability. The animal:

- **Hits** on ~33% of all trials, but **misses** on ~12% — failing to respond to Go stimuli with some regularity.
- **False alarms** on ~15% of trials — licking impulsively on Nogo stimuli.
- **Aborts** nearly 1 in 5 trials by licking during the inter-trial interval (`answer_window = 0`).

This variability is not random. Animals tend to fluctuate between structured **epochs** of behavior — stretches of good performance (high hits, low false alarms) alternating with periods of disengagement (many misses, many aborted trials) or impulsivity (many false alarms). This is precisely the kind of structure that hidden Markov models (HMMs) are designed to uncover.

---

## Connection to the Hulsey et al. (2024) Paper

The Hulsey et al. paper applied **GLM-HMMs** (Hidden Markov Models with Generalized Linear Model emissions) to a similar behavioral dataset and found that mice fluctuate between discrete **performance states**:

- **Optimal** — responses correctly guided by the stimulus
- **Sub-optimal** — systematic biases (e.g., always licking left, or avoiding one stimulus)
- **Disengaged** — not responding to stimuli at all

They showed that these states can be predicted by **arousal measures** (pupil diameter, locomotion speed, facial motion energy), with optimal performance occurring at intermediate arousal levels — consistent with the classic Yerkes-Dodson law.

The present dataset is well-suited for a similar analysis. The trial-by-trial state sequence, session boundaries, and weekend break markers all provide the structure needed to fit and interpret an HMM — and to study how performance states evolve over time and across training.
