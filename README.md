**Indian Legal LLM**  
:scales: A state-of-the-art large language model fine-tuned on Indian legal statutes, case law, and Q&A datasets.

[![Model Card](https://img.shields.io/badge/Model-Indian%20Legal%20LLM-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](#)

---

## 🚀 Features

- **Comprehensive Coverage**: Trained on IPC, CrPC, BNS, BSA, IEA, Constitution, and more.
- **Case Law Retrieval**: Context-aware retrieval of Supreme Court and High Court judgments.
- **Statute Summarization**: Generates concise summaries of complex legal provisions.
- **Q&A Support**: Answer legal queries with citations and section references.
- **Multi-Task**: Supports drafting pleadings, compliance checklists, and legislative analysis.

## 📚 Data Sources

| Source                                   | Type           | Coverage                                     |
|------------------------------------------|----------------|----------------------------------------------|
| ILDC                                     | Case Law       | 35K+ Supreme Court Judgments                 |
| NJDG                                     | Case Metadata  | 30+ Crore Orders & Judgments                 |
| India Code Portal                        | Statutes       | Central & State Acts + Rules & Notifications |
| IndicLegalQA                             | QA Pairs       | 1.2K+ Expert-Reviewed Q&A                    |
| InLegalBERT Pretraining Data             | Documents      | 5.4M+ Court Documents                        |

> _Tip_: See [DATA_SOURCES.md](./DATA_SOURCES.md) for full dataset list and ingestion scripts.

## 🏗️ Architecture

```text
+----------------+     +----------------------+     +----------------+
|  Raw Text Data | --> | Preprocessing & OCR  | --> | Tokenization   |
+----------------+     +----------------------+     +----------------+
                                   |                             |
                                   v                             v
                         +----------------------+     +----------------+
                         | Retrieval-Augmented |     | Fine-tuning on |
                         |  Generation (RAG)    |     |  Legal Corpora |
                         +----------------------+     +----------------+
                                   |
                                   v
                          +-------------------+
                          | Deployed as API   |
                          +-------------------+
```

## 💾 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/indian-legal-llm.git
   cd indian-legal-llm
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained weights**
   ```bash
   bash scripts/download_weights.sh
   ```

## ⚙️ Usage

### Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-org/indian-legal-llm")
model = AutoModelForCausalLM.from_pretrained("your-org/indian-legal-llm")

prompt = "Explain Section 375 of the IPC."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### RAG Retrieval

```bash
python scripts/run_rag.py --query "Landmark SC judgment on property rights"
```

## 📈 Training

- **Prepare data**: Place raw `.jsonl` files under `data/raw/`.
- **Preprocess**:
  ```bash
  python scripts/preprocess.py --input data/raw --output data/processed
  ```
- **Train**:
  ```bash
  python scripts/train.py \
    --model_name_or_path gpt-neo-1.3B \
    --train_file data/processed/train.jsonl \
    --validation_file data/processed/val.jsonl \
    --output_dir outputs/indian-legal-llm
  ```
- **Evaluate**:
  ```bash
  python scripts/evaluate.py --predictions outputs/preds.jsonl --references data/processed/test.jsonl
  ```

## 📊 Evaluation Metrics

| Metric          | Description                         |
|-----------------|-------------------------------------|
| EM (Exact Match)| Strict text match of answers        |
| F1 Score        | Token-level overlap                 |
| BLEU            | N-gram precision for summaries      |
| Rouge-L         | Longest common subsequence measure  |

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open a Pull Request.

Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## 📬 Contact

Maintainer: [Your Name](mailto:your.email@example.com)  
Project Repo: [github.com/your-org/indian-legal-llm](https://github.com/your-org/indian-legal-llm)
