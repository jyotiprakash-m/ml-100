# 📘 Online Learning in Machine Learning

## 🔷 What is Online Learning?

**Online learning**, also called _incremental learning_, is a machine learning paradigm where the model learns **one observation at a time or in small batches** as data arrives.

It is ideal for:

- Large datasets that cannot fit in memory.
- Streaming or real-time data.
- Situations where the data distribution changes over time (_concept drift_).

---

## 🚀 Why Use Online Learning?

- ✅ Works on streaming or big data.
- ✅ Continuously updates the model as new data comes.
- ✅ Adaptable to changing environments.

---

## 🛠️ Supported Algorithms

Not all models can learn incrementally.
Popular online-learning models in scikit-learn include:

- `SGDClassifier`, `SGDRegressor`
- `Perceptron`
- `PassiveAggressiveClassifier`
- `MiniBatchKMeans`
- `GaussianNB` (supports `partial_fit`)

These models implement the `.partial_fit()` method.

---

## 🔷 Workflow

```mermaid
graph TD
    A[Stream or Chunked Data] --> B[Preprocess Each Chunk]
    B --> C[Call .partial_fit()]
    C --> D[Model Updated]
    D --> C
```

## 📊 Advantages

- ✅ Works with large/streaming data.
- ✅ Can adapt to changing data distributions.
- ✅ No need to retrain from scratch for new data.

## ⚠️ Limitations

- ⚠️ Only works with certain algorithms.
- ⚠️ Sensitive to the order and quality of incoming data.
- ⚠️ Careful tuning of hyperparameters (like learning rate) is required.

---

## 📚 References

- [scikit-learn: Incremental learning](https://scikit-learn.org/stable/whats_new/v1.7.html#id21)
- [Wikipedia: Online machine learning](https://en.wikipedia.org/wiki/Online_machine_learning)
