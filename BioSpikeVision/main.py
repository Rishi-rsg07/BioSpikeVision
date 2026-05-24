from src.pipeline import BioSpikePipeline

if __name__ == "__main__":
    # Standard 0 opens the default integrated webcam
    pipeline = BioSpikePipeline(video_source=0)
    pipeline.run()