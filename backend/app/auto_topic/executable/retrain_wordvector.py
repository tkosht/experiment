import joblib
import typer
from tqdm import tqdm

from app.auto_topic.component.logger import Logger


g_logger = Logger(logger_name="retrain_wordvector")


def log_info(*args, **kwargs):
    g_logger.info(*args, **kwargs)


def main(
    batch_size: int = 10000,
    pipe_file: str = "data/pipe_wikivec.gz",
    sentence_file: str = "tidy_sentences.gz",
):
    log_info("Start", "retrain_wordvector")
    tidy_sentences = joblib.load(sentence_file)
    pipe_wikivec = joblib.load(pipe_file)

    # NOTE: around 5 hours (303m 54.5s)
    for idx, offset in enumerate(tqdm(range(0, len(tidy_sentences), batch_size))):
        X = tidy_sentences[offset : offset + batch_size]
        print(offset, offset + batch_size)
        pipe_wikivec.fit([X])

    # NOTE: around 2m = 1m55.0s
    log_info("Start", "dumping retrained model")
    joblib.dump(pipe_wikivec, "data/pipe_wikivec.retrained.gz", compress=("gzip", 3))
    log_info("End", "dumping retrained model")

    log_info("End", "retrain_wordvector")


if __name__ == "__main__":
    typer.run(main)
