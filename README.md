# AI Recommendation System (Hybrid, Production-Oriented)

## Problem Statement
Build an event-driven recommendation system for an e-commerce platform that ranks products for users using implicit behavioral signals such as views, cart additions, and purchases.

## Scope (Phase 1)
- Data ingestion and cleaning
- Popularity-based recommendation baseline
- Offline evaluation using time-based splits

## Why This Project
Most recommendation demos focus only on models.
This project focuses on data realism, baselines, evaluation, and system thinking.

## Data Observations
- **Interaction data is highly sparse**: 500K interactions across 155K users × 19K items = 0.0162% density
- **Item interactions follow a long-tail distribution**: Top 10% of items account for 52.9% of interactions; majority of items get <20 interactions
- **User activity is heavily skewed**: Small group of power users (top 10 users account for 0.3% of interactions) with most users having 1-5 interactions

These patterns are typical for e-commerce and require careful handling:
- Sparsity → need collaborative filtering and regularization
- Long-tail → popularity baseline will be strong; need techniques to surface niche items
- User skew → interaction weighting and normalization are critical

### Model Training
- Train models using scripts in `src/models/`

### Evaluation
- Evaluate model performance using `src/evaluation/`

## Notebooks

Jupyter notebooks for exploratory data analysis and experiments are stored in `notebooks/`

```bash
jupyter notebook
```

## Development

Format code with Black:
```bash
black src/
```

Run tests:
```bash
pytest
```

Lint code:
```bash
flake8 src/
```

## License

[Add your license here]
