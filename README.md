# 🤖 KAN Robot control 

This work investigates the performance of **Kolmogorov–Arnold Networks** (KAN) compared to standard Multi-Layer Perceptrons (MLP) for estima the final 3D position of markers on a soft robotic system undergoing mechanical deformation. Both models are trained on static actuation data and evaluated on unseen configurations, with a focus on generalization across actuation subspaces. As an extension, we explore a sequential variant of KAN (KAN-RNN), incorporating temporal dynamics through recurrent processing. Experimental results highlight the superior accuracy of KAN over MLP in the static setting, while the KAN-RNN shows potential but remains less competitive in its current form.

---

### 🔍 Key Findings

- **KAN outperforms MLP** in the static setting in terms of accuracy.
- **KAN-RNN** introduces temporal modeling, but is currently **less competitive**.
- Results suggest **promising directions for further development** of KAN-based sequential models in soft robotics.

---

### 📂 Repository Structure

- **`dataset/`** – Contains the data used for training and testing the models.
- **`models/`** – Stores all models used in the project. Models labeled as **best** are those obtained after fine-tuning. The others are used for Continual Learning (CL) tasks.
- **`plots/`** – Includes all generated plots and visualizations.
- **`presentation/`** – Contains the project presentation, with a detailed explanation of the results obtained.
- **`src/`** – Holds the implementation of the models and core logic.

In addition to the folders listed above, the repository also includes **main files** that differ depending on the specific task of the *sistema* project.

