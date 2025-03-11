import logging


def create_logger(log_file: str):
    # 建立 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 建立終端輸出 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 建立檔案輸出 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 設定 formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 將 handler 加入 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 使用範例
logger = create_logger("app.log")
logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")
