# Meta Data Extraction

**Meta Data Extraction** is a Python module designed to extract and store meta-features from the results of knowledge base experiments using the **meta features extraction module (process features)**. These meta-features are compiled into a meta-dataset, which is then used to train a meta-model that predicts the most suitable calibration method for a given set of true labels and predicted probabilities from the calibration set.

---

## Usage

To extract and store meta-features from knowledge base experiment results, run the following script:

```bash
python -m meta_data_extraction.meta_data_extraction_script
```

- To specify the source table for meta-feature extraction, update the following line in the script:

```python
process_meta_features(table=KnowledgeBaseExperiment_V2)
```

Replace `KnowledgeBaseExperiment_V2` with the needed table.
