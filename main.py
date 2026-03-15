from dotenv import load_dotenv

from macro_stress_pipeline.pipeline import run

load_dotenv()

if __name__ == "__main__":
    run()
