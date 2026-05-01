# STARMAP: Structural Transcriptomic Analysis of Residue-level Mutation-Associated Pathways

STARMAP is a computational framework for integrating protein structure, mutation data, transcriptional networks, and drug response to identify functionally relevant regions of proteins and their downstream effects.

This repository contains code associated with the STARMAP manuscript (currently in preparation), including reproducible analysis workflows and example pipelines.

---

## Repository Contents

### 1. Example Pipeline (TP53)
A minimal working pipeline is provided for **TP53** to demonstrate how STARMAP can be run end-to-end. This includes:

- Structural processing and feature generation  
- Dimensionality reduction and clustering (Regions of Functional Interest, RFIs)  
- Transcriptional regulatory network (TRN) association analysis  
- Drug response modeling  

This serves as a reference implementation for applying STARMAP to additional proteins.

---

### 2. Reproducing Manuscript Analyses
Code used to generate the analyses and figures from the manuscript is included. These scripts:

- Reproduce key statistical analyses  
- Generate plots and summary metrics  
- Recreate figure panels from the paper  

Note that some analyses rely on large intermediate datasets and may require adaptation depending on compute environment.

---

### 3. Data Availability

Due to the size of the full STARMAP outputs, they are not hosted directly in this repository.

Full processed outputs, including:
- Protein-level scores  
- Cluster-level annotations  
- TRN associations  
- Drug response metrics  

are available at:

https://starmap.unc.edu

---

## Getting Started

### Requirements
The pipeline relies on standard scientific Python libraries, including:

- pandas  
- numpy  
- scikit-learn  
- matplotlib / seaborn  
- scipy  

Additional dependencies may be required for specific modules.

### Running the Example
To run the TP53 pipeline:

1. Navigate to the TP53 example directory  
2. Follow the provided script/notebook instructions  
3. Ensure required input data paths are configured  

---

## License

This project is released under the **STARMAP Academic Use License**.

### Summary
- Free for non-commercial academic and research use  
- Modification and redistribution allowed for academic purposes  
- Commercial use is strictly prohibited without explicit permission  

For commercial licensing or other inquiries, please contact the authors.

---

## Citation

The STARMAP manuscript is currently in preparation.  
A citation will be provided upon publication.

---

## Contact

For questions, collaboration inquiries, or licensing requests, please reach out to the repository authors.
