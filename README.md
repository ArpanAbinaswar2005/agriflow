# AgriFlow

Two things break India's agricultural supply chain. Farmers load trucks without knowing where their crop will get the best price. Distant buyers won't commit to a purchase without physically inspecting the produce. AgriFlow fixes both.

## What it does

**Pipeline A — Price Forecasting**

Takes five years of Agmarknet price records and pairs them with IMD district-level weather forecasts inside a dual-encoder LSTM. The weather encoder matters because price history alone can't see a drought forming three weeks before arrivals drop. The model produces 7, 14, and 30-day price and supply deficit forecasts per commodity-mandi pair — before the truck leaves the farm.

**Pipeline B — Quality Grading**

Stage 1 runs on the farmer's phone. They shoot a 10-second video of the crate and the system samples frames at random, so they can't stage a clean shot while hiding spoiled produce at the bottom. This gives a provisional grade used only for routing.

Stage 2 runs at the mandi gate. A multi-modal network takes camera images, transit temperature logs, and weight-density readings and issues the verified grade that actually sets the price. Because settlement happens at the gate, there's no payoff from faking the farm-side video.

**Routing Engine**

Combines the price forecast, supply deficit score, quality grade, and estimated transport cost into a ranked list of mandi recommendations. The farmer gets a destination before loading, not after.

## Architecture

```
Agmarknet prices ──┐
                   ├─► Dual-encoder LSTM ──► Price forecast + Deficit score ──┐
IMD weather data ──┘                                                           │
                                                                               ▼
Farmer video ───────► MobileNetV3 (Stage 1) ──► Provisional grade ──► Routing Engine ──► Ranked mandis
                                                                               ▲
Fixed-rig camera ──┐                                                           │
Temp/humidity ─────├─► Late-fusion network (Stage 2) ──► Verified grade ──────┘
Weight/density ────┘
```

## Datasets

| Dataset | Pipeline | Use |
|---|---|---|
| Agmarknet 2019–2024 | A | Price and volume forecasting |
| IMD district weather | A | Weather history and forecast input |
| NASA POWER (gridded) | A | Gap-filling by mandi coordinates |
| PlantVillage | B | Disease-aware pretraining |
| Fruits-360 | B | Colour-texture feature learning |
| Custom Odisha mandi dataset | B | Fine-tuning both stages |

## Models

- Price forecasting: dual-encoder LSTM with cross-attention (price encoder: 2 layers, hidden dim 128 / weather encoder: 2 layers, hidden dim 64)
- Stage 1 grading: MobileNetV3-Large backbone, majority vote across random video frames
- Stage 2 grading: ResNet-50 visual encoder + masked-modality late-fusion head
- Explainability: Grad-CAM saliency overlays on the visual branch for buyer verification

## Project structure

```
agriflow/
├── pipeline_a/
│   ├── data_loader.py        # Agmarknet + IMD data ingestion
│   ├── model.py              # Dual-encoder LSTM architecture
│   └── train.py              # Training script
├── pipeline_b/
│   ├── model_stage1.py       # MobileNetV3 farm-side grader
│   └── model_stage2.py       # ResNet-50 + late-fusion gate grader
├── routing/
│   └── engine.py             # Mandi scoring and ranking
├── models/                   # Saved model checkpoints
└── main.py                   # End-to-end inference
```

Training runs on Google Colab with a T4 GPU. See each pipeline folder for training instructions.

## Status

Active development. Built for DeLTA 2026 (Registration No. DeLTA26 091).

Pilot deployment with an Odisha mandi board is planned for real-world data collection and validation.
