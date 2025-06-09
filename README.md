# 🧠 Adaptive Logic-of-Thought (A-LoT)

This repository contains the implementation and evaluation of **Adaptive Logic-of-Thought (A‑LoT)**, a dynamic prompting framework designed to enhance large language model reasoning by adapting logical augmentation to task complexity.

## 🔍 Overview

**A‑LoT** intelligently classifies reasoning tasks based on their complexity—**Low**, **Medium**, or **High**—and applies the optimal level of logical prompting accordingly:

- **Low** → Chain-of-Thought (CoT) only  
- **Medium** → CoT + targeted logical hints  
- **High** → Full Logic-of-Thought (LoT) augmentation

This adaptive strategy boosts accuracy on harder problems and reduces overengineering on easier ones.

---

## 📊 Results

| Complexity | Metric   | CoT  | LoT  | A‑LoT (Ours) |
|------------|----------|------|------|---------------|
| Low        | Accuracy | 90.0 | 92.0 | **90.0**      |
| Medium     | Accuracy | 70.0 | 75.0 | **78.0**      |
| High       | Accuracy | 60.0 | 78.0 | **78.0**      |
| **Overall**| Accuracy | 77.0 | 82.0 | **84.0**      |

🟢 A‑LoT delivers **the highest overall accuracy** while minimizing unnecessary computation for simpler tasks.

---

## 📁 Contents

- `a_lot.py` – Core logic for complexity scoring, adaptive prompting, and puzzle solving.
- `examples/` – Sample puzzles used for evaluation (e.g., arithmetic, riddles, logic).
- `benchmark_results.csv` – Detailed performance data.
- `requirements.txt` – Python dependencies.

---

## 📚 Citation & References

- Wei, J. et al. “Chain-of-Thought Prompting Elicits Reasoning in LLMs”, *arXiv:2201.11903*  
- Chen, A. et al. “Logic-of-Thought: Injecting Logic into Contexts for Full Reasoning in LLMs”, *arXiv:2409.17539*  
- Lee, A. & Patel, P. “ComplexityNet: Increasing Language Model Inference Efficiency by Learning Task Complexity”, *arXiv:2402.01234*

---

🔗 GitHub: [https://github.com/akhilender-bongirwar/LCS2022011_Research_Paper](https://github.com/akhilender-bongirwar/LCS2022011_Research_Paper)
