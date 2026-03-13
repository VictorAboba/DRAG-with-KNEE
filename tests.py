import re

transfer_pattern = re.compile(r"(\S)-\s+(\S)|(\S)\s+-(\S)")

text = "smt- h, dasdasdasd, smt - h, sdsaadasd smt-h, smt -h"

print(re.sub(transfer_pattern, r"\1\2\3\4", text))
