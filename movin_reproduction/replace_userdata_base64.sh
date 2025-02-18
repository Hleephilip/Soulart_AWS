#!/bin/bash

# Define input and output file
INPUT_FILE="user_data_script.sh"
OUTPUT_FILE="user_data_script.sh.b64"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found!"
    exit 1
fi

# Encode file to base64 and save to output file
base64 -w0 "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Encoded file saved to $OUTPUT_FILE"