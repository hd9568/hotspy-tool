#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1
output_file="demangle.txt"

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

if [ -f "$output_file" ]; then
    rm "$output_file"
fi

while IFS= read -r line; do
    demangled=$(c++filt "$line")
    echo "$demangled" >> "$output_file"
done < "$input_file"

echo "Demangling complete. Output saved to '$output_file'."

