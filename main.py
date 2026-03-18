from dotenv import load_dotenv

from macro_stress_ml.optimizer import run as run_optimizer
from macro_stress_pipeline.pipeline import run as run_pipeline

load_dotenv()

if __name__ == "__main__":
    run_pipeline()
    run_optimizer()
