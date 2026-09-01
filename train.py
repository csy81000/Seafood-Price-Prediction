"""날씨와 어획량으로 수산물 단가를 예측하는 회귀 모델 학습 스크립트."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = PROJECT_DIR / "data" / "train.csv"
DEFAULT_TEST_FILE = PROJECT_DIR / "data" / "test.csv"

WEATHER_COLUMNS = [
    "평균기압(hPa)",
    "평균 상대습도(%)",
    "평균 기온(°C)",
    "평균 수온(°C)",
    "평균 유의 파고(m)",
]
CATCH_COLUMNS = [
    "갈치(상) 어획량",
    "갈치(중) 어획량",
    "갈치(하) 어획량",
    "고등어(상) 어획량",
    "고등어(중) 어획량",
    "고등어(하) 어획량",
    "갈고등어 어획량",
    "갑오징어 어획량",
    "오징어A 어획량",
]
PRICE_COLUMNS = [
    "갈치(상) 단가",
    "갈치(중) 단가",
    "갈치(하) 단가",
    "고등어(상) 단가",
    "고등어(중) 단가",
    "고등어(하) 단가",
    "갈고등어 단가",
    "갑오징어 단가",
    "오징어A 단가",
]
PRODUCT_KEYS = [
    "hairtail-high",
    "hairtail-medium",
    "hairtail-low",
    "mackerel-high",
    "mackerel-medium",
    "mackerel-low",
    "scad",
    "cuttlefish",
    "squid-a",
]


@dataclass(frozen=True)
class TrainingPreset:
    hidden_units: tuple[int, ...]
    dropout_rate: float
    learning_rate: float
    epochs: int
    batch_size: int


PRESETS = {
    "baseline": TrainingPreset(
        hidden_units=(128, 64),
        dropout_rate=0.0,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
    ),
    "tuned": TrainingPreset(
        hidden_units=(128, 64, 32, 16),
        dropout_rate=0.3,
        learning_rate=0.001,
        epochs=300,
        batch_size=16,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="수산물 9개 품목의 단가 예측 모델을 학습하고 평가합니다."
    )
    parser.add_argument(
        "--preset", choices=PRESETS, default="baseline", help="사용할 모델 구성"
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="기본값: outputs/<preset>",
    )
    parser.add_argument("--epochs", type=int, help="프리셋의 epoch 수를 덮어씁니다.")
    parser.add_argument(
        "--batch-size", type=int, help="프리셋의 batch size를 덮어씁니다."
    )
    parser.add_argument("--seed", type=int, default=12321)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--skip-plots", action="store_true", help="학습/예측 그래프를 생성하지 않습니다."
    )
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_datasets(train_file: Path, test_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (train_file, test_file):
        if not path.is_file():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    required_columns = set(WEATHER_COLUMNS + CATCH_COLUMNS + PRICE_COLUMNS)

    for label, data in (("train", train_data), ("test", test_data)):
        missing_columns = sorted(required_columns.difference(data.columns))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{label} 데이터에 필요한 열이 없습니다: {missing}")
    return train_data, test_data


def build_model(input_size: int, preset: TrainingPreset) -> keras.Sequential:
    layers: list[keras.layers.Layer] = [keras.layers.Input(shape=(input_size,))]
    for index, units in enumerate(preset.hidden_units):
        layers.append(keras.layers.Dense(units, activation="relu"))
        if preset.dropout_rate and index > 0:
            layers.append(keras.layers.Dropout(preset.dropout_rate))
    layers.append(keras.layers.Dense(1, activation="linear"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=preset.learning_rate),
        loss="mse",
    )
    return model


def prepare_product_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    catch_column: str,
    price_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler] | None:
    feature_columns = WEATHER_COLUMNS + [catch_column]
    train_rows = train_data[feature_columns + [price_column]].dropna()
    test_rows = test_data[feature_columns + [price_column]].dropna()
    train_rows = train_rows[train_rows[price_column] > 0]
    test_rows = test_rows[test_rows[price_column] > 0]

    if len(train_rows) < 10 or len(test_rows) < 10:
        return None

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_rows[feature_columns])
    x_test = scaler.transform(test_rows[feature_columns])
    y_train = train_rows[price_column].to_numpy(dtype=np.float32)
    y_test = test_rows[price_column].to_numpy(dtype=np.float32)
    return x_train, y_train, x_test, y_test, scaler


def save_plots(
    histories: dict[str, dict[str, list[float]]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 3, figsize=(18, 12))
    for axis, (price_column, history) in zip(axes.flat, histories.items()):
        axis.plot(history["loss"], label="train loss")
        axis.plot(history["val_loss"], label="validation loss")
        axis.set_title(price_column)
        axis.set_xlabel("epoch")
        axis.set_ylabel("MSE")
        axis.grid(True)
        axis.legend()
    for axis in list(axes.flat)[len(histories):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(plot_dir / "loss_history.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(18, 12))
    for axis, (price_column, (actual, predicted)) in zip(axes.flat, predictions.items()):
        axis.plot(actual, label="actual", linestyle="--", linewidth=1)
        axis.plot(predicted, label="predicted", linewidth=1)
        axis.set_title(price_column)
        axis.set_xlabel("sample")
        axis.set_ylabel("price")
        axis.grid(True)
        axis.legend()
    for axis in list(axes.flat)[len(predictions):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(plot_dir / "prediction_comparison.png", dpi=150)
    plt.close(figure)


def run_training(args: argparse.Namespace) -> pd.DataFrame:
    preset = PRESETS[args.preset]
    epochs = args.epochs if args.epochs is not None else preset.epochs
    batch_size = args.batch_size if args.batch_size is not None else preset.batch_size
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs와 batch size는 1 이상이어야 합니다.")

    output_dir = args.output_dir or PROJECT_DIR / "outputs" / args.preset
    model_dir = output_dir / "models"
    scaler_dir = output_dir / "scalers"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    configure_reproducibility(args.seed)
    train_data, test_data = load_datasets(args.train_file, args.test_file)
    train_data = train_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    metrics: list[dict[str, float | int | str]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for product_key, catch_column, price_column in zip(
        PRODUCT_KEYS, CATCH_COLUMNS, PRICE_COLUMNS
    ):
        prepared = prepare_product_data(
            train_data, test_data, catch_column, price_column
        )
        if prepared is None:
            print(f"[skip] {price_column}: 학습 또는 테스트 데이터가 부족합니다.")
            continue

        x_train, y_train, x_test, y_test, scaler = prepared
        model = build_model(x_train.shape[1], preset)
        print(f"[train] {price_column} ({len(y_train)} train / {len(y_test)} test)")
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=args.verbose,
        )
        predicted = model.predict(x_test, verbose=0).reshape(-1)

        mae = float(mean_absolute_error(y_test, predicted))
        mse = float(mean_squared_error(y_test, predicted))
        mape = float(np.mean(np.abs((y_test - predicted) / y_test)) * 100)
        metrics.append(
            {
                "product": price_column,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "MAE": mae,
                "MSE": mse,
                "MAPE_percent": mape,
            }
        )
        histories[price_column] = {
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
        }
        predictions[price_column] = (y_test, predicted)

        model.save(model_dir / f"{product_key}-price-model.keras")
        joblib.dump(scaler, scaler_dir / f"{product_key}-scaler.joblib")
        print(f"[result] MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape:.2f}%")
        del model
        keras.backend.clear_session()

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "preset": args.preset,
        "preset_config": asdict(preset),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": args.seed,
        "features": WEATHER_COLUMNS,
        "catch_columns": CATCH_COLUMNS,
        "targets": PRICE_COLUMNS,
        "product_keys": PRODUCT_KEYS,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.skip_plots and histories:
        save_plots(histories, predictions, output_dir)
    return metrics_frame


def main() -> None:
    args = parse_args()
    results = run_training(args)
    if results.empty:
        print("학습할 수 있는 품목이 없습니다.")
    else:
        print("\n수산물별 예측 성능")
        print(results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
