# README

## Multi-Center Medication Recommendation via Adaptive Graph-Based Modeling (AGBM)

This README provides the instructions required to reproduce and verify the experimental results presented in the research article:

**"Multi-Center Medication Recommendation via Adaptive Graph-Based Modeling"**

Please follow the steps below to execute the proposed AGBM framework.

### Step 1: Download the [eICU](https://eicu.mit.edu/) Dataset

1. Apply for access to the **[eICU](https://eicu.mit.edu/) Collaborative Research Database**.
2. Download the dataset after obtaining the required permissions.
3. Extract the downloaded files.
4. Place the following CSV files into the directory:

```text
data/eicu/raw/
```

Required files:

* `diagnosis.csv`
* `medication.csv`
* `patient.csv`
* `treatment.csv`

---

### Step 2: Data Pre-processing

Run the following script to preprocess and clean the raw [eICU](https://eicu.mit.edu/) dataset:

```bash
python data/processing-eicu.py
```

---

### Step 3: Hospital Filtering

Execute the following script to remove hospitals with too few patient records:

```bash
python data/filter_hospitals.py
```

---

### Step 4: Data Splitting

After preprocessing, the processed dataset will be generated in:

```text
data/eicu/handled/
```

The dataset is randomly divided into training, validation, and testing sets over **five independent runs** using the following random seeds:

```text
[42, 43, 44, 45, 46]
```

The reported experimental results in the manuscript are obtained by averaging the performance across these five runs.

---

### Step 5: Execute the AGBM Framework

Run the proposed Adaptive Graph-Based Modeling (AGBM) framework using:

```bash
python AGBM.py
```

The script automatically trains the model and evaluates its performance on the generated data splits.

---

## Notes

* Ensure that all required Python packages and dependencies are installed before executing the scripts.
* The directory structure should remain unchanged to ensure the scripts locate the required files correctly.
* All experiments reported in the manuscript were conducted using the preprocessing pipeline and random seeds described above.
* Following the above procedure should enable reviewers and readers to reproduce the experimental results presented in the manuscript.
# AGBM
