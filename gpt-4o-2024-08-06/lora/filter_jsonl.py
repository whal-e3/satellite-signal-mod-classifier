#!/usr/bin/env python3
import json

# --- Configuration ---
# The large dataset file you already created
INPUT_FILE = 'dataset_all.jsonl' 
# The new, filtered file to be created
OUTPUT_FILE = 'dataset_clean_training0704.jsonl'
# The keyword to filter by
FILTER_KEYWORD = '_training0704'

# --- Main Script ---
print(f"Filtering '{INPUT_FILE}' for lines containing '{FILTER_KEYWORD}'...")
lines_written = 0

try:
    with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w') as outfile:
        # Process each line from the input file
        for line in infile:
            data = json.loads(line)
            
            # Find the user message that contains the list of images
            image_list_message = None
            for msg in data['messages']:
                # The message with images has a list as its content
                if isinstance(msg.get('content'), list):
                    image_list_message = msg
                    break
            
            if not image_list_message:
                continue

            # Check if any URL in the list contains the keyword
            found_match = False
            for image_item in image_list_message['content']:
                if 'image_url' in image_item and FILTER_KEYWORD in image_item['image_url']['url']:
                    # If a match is found, write the original line to the new file
                    outfile.write(line)
                    lines_written += 1
                    found_match = True
                    # Break the inner loop since we only need one match per line
                    break 
            
    print(f"\n✅ Success! Created '{OUTPUT_FILE}' with {lines_written} matching entries.")

except FileNotFoundError:
    print(f"❌ Error: Input file '{INPUT_FILE}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")