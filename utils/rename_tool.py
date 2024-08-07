# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import re
import os


def rename_classes_in_file(file_path, pattern, replacement):
    with open(file_path, "r") as file:
        content = file.read()

    new_content = re.sub(pattern, replacement, content)

    with open(file_path, "w") as file:
        file.write(new_content)


def main():
    directory = "D:/fl.lutz/FAST/FAST-OAD/FAST-OAD-CS23-HE/src/fastga_he/models"
    pattern = r"\bsizing.*?_cg\b"
    replacement = r"\g<0>_x"

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                if filename.startswith("test_") or filename.startswith("sizing_"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, "r") as file:
                        content = file.read()

                    modified_content = re.sub(pattern, replacement, content)

                    if modified_content != content:
                        with open(file_path, "w") as file:
                            file.write(modified_content)
                        print(f"Modified classes in {file_path}")


if __name__ == "__main__":
    main()
