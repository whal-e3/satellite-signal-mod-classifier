#!/usr/bin/env python3
import json
import os

# --- Configuration ---
INPUT_FILE = 'dataset_grouped.jsonl'
OUTPUT_FILE = 'finetuning_dataset.jsonl'

# Static parts of the conversation that will be the same for every entry
SYSTEM_PROMPT = "You are a modulation classification assistant."
USER_TEXT_PROMPT = (
    "Given four images showing a radio signal's characteristics (constellation diagram, "
    "time-domain graph, frequency-domain graph, waterfall diagram), identify the modulation "
    "type from: WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, GMSK, CW, CSS, BFSK, GFSK. "
    "Output the modulation type only."
)

# --- Main Script ---
print(f"Reading from '{INPUT_FILE}' and creating '{OUTPUT_FILE}'...")
entry_count = 0

try:
    # Open both files safely
    with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w') as outfile:
        # Process each line from the input file
        for line in infile:
            data = json.loads(line)
            image_urls = data['image_urls']
            source_directory = data['source_directory']

            # Extract the modulation type (the label) from the directory path.
            # e.g., 'mother_folder/8PSK/10db' -> '8PSK'
            path_parts = os.path.normpath(source_directory).split(os.path.sep)
            if len(path_parts) < 2:
                print(f"Warning: Skipping malformed path '{source_directory}'")
                continue
            assistant_label = path_parts[-2]

            # Build the list of image URL objects for the user's message
            user_image_content = []
            for url in image_urls:
                user_image_content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

            # Assemble the complete list of messages for this training example
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEXT_PROMPT},
                {"role": "user", "content": user_image_content},
                {"role": "assistant", "content": assistant_label}
            ]

            # Create the final JSON object for this line
            final_json_obj = {"messages": messages}

            # Write the complete JSON object to the new file, followed by a newline
            outfile.write(json.dumps(final_json_obj) + '\n')
            entry_count += 1

    print(f"\n✅ Successfully created '{OUTPUT_FILE}' with {entry_count} entries.")

except FileNotFoundError:
    print(f"❌ Error: Input file '{INPUT_FILE}' not found. Make sure it's in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")