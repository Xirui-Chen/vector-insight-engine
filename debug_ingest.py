# debug_ingest.py

from dotenv import load_dotenv
from ingest import ingest_text

def main() -> None:
    """
    Tiny debug script to see where ingest_text hangs.
    It prints clear progress messages.
    """
    print("Step 1: Loading .env with load_dotenv('.env') ...")
    loaded = load_dotenv(".env")
    print("Step 1 done. dotenv loaded =", loaded)

    sample_text = (
        "Machine learning models depend on the quality and coverage of their training data. "
        "Poor inputs lead to unstable predictions and biased outcomes. "
        "Robust data pipelines improve accuracy and interpretability for high impact decisions. "
        "Vector Insight Engine is built to help analysts turn messy documents into searchable insights."
    )

    print("Step 2: Calling ingest_text(...) now ...")
    try:
        written = ingest_text(
            sample_text,
            project="debug-project",
            document_name="debug-manual",
        )
        print("Step 3: ingest_text returned, written chunks =", written)
    except Exception as e:
        print("Step 3: ingest_text raised an exception:")
        import traceback
        traceback.print_exc()

    print("Debug script finished.")

if __name__ == "__main__":
    main()
