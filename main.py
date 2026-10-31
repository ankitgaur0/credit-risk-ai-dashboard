"""Build step: raw CSVs -> clean dataset -> trained model + account scores.

Run:  uv run python main.py
"""

from src import ingestion, model


def main():
    ingestion.main()
    model.main()


if __name__ == "__main__":
    main()
