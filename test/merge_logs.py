import os
import re
from datetime import datetime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_timestamp(line: str):
    match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]", line)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
    return None


def merge_logs_by_timestamp(input_dir: str, output_file: str):
    log_entries = []

    for dirname in os.listdir(os.path.abspath(input_dir)):
        dirpath = os.path.join(input_dir, dirname)
        if os.path.isfile(dirpath):
            continue

        if os.path.exists(os.path.join(dirpath, output_file)):
            continue

        for filename in os.listdir(dirpath):
            if not filename.endswith(".log"):
                continue

            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r") as f:
                for line in f:
                    ts = parse_timestamp(line)
                    if ts:
                        log_entries.append((ts, line.strip()))
                    else:
                        # 沒有 timestamp 的行視為前一行的延伸
                        if log_entries:
                            log_entries[-1] = (
                                log_entries[-1][0],
                                log_entries[-1][1] + "\n" + line.strip(),
                            )

        # 時間排序
        log_entries.sort(key=lambda x: x[0])

        # 寫入合併結果
        with open(os.path.join(dirpath, output_file), "w") as out:
            for _, line in log_entries:
                out.write(line + "\n")


# 使用方式
merge_logs_by_timestamp("./logs", "megred.log")
