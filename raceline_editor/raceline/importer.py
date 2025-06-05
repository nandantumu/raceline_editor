import csv


def import_raceline_csv(file_path: str) -> list[list[str]]:
    """
    Imports raceline data from a CSV file.
    It identifies a header row (e.g., commented with '#' and containing ',') and normalizes it.
    Returns a list of lists, where the first list is the cleaned header,
    and subsequent lists are data rows.
    Raises ValueError if a suitable header is not found or if the file is empty/malformed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        # Let this propagate or handle as per application's error strategy
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")

    if not lines:
        # Return empty list or raise error, consistent with how no header is handled
        raise ValueError(f"CSV file {file_path} is empty.")

    header_idx = -1
    raw_header_content_after_hash = ""

    for i, line_content in enumerate(lines):
        stripped_line = line_content.strip()
        # Header identification: starts with '#' and contains a common delimiter (',' or ';')
        if stripped_line.startswith("#") and ("," in stripped_line or ";" in stripped_line):
            header_idx = i
            raw_header_content_after_hash = stripped_line.lstrip("#")
            break
        # Optional: Fallback to first non-empty line if no '#'-prefixed header is found
        # For now, sticking to the stricter reference logic.
        # If a more lenient approach is needed, this is where it would go.
        # elif header_idx == -1 and stripped_line: # First non-empty line as a fallback
        #     header_idx = i
        #     raw_header_content_after_hash = stripped_line
        #     break

    if header_idx == -1:
        raise ValueError(
            f"No header line found in {file_path}. "
            "The importer expects a header line starting with '#' and containing ',' or ';', "
            "e.g., '#col1,col2,col3...' or '#col1;col2;col3...'"
        )

    # Determine delimiter from the raw header content
    comma_present = ',' in raw_header_content_after_hash
    semicolon_present = ';' in raw_header_content_after_hash

    if semicolon_present and not comma_present:
        delimiter = ';'
    elif comma_present and not semicolon_present:
        delimiter = ','
    elif semicolon_present and comma_present:
        # If both are present, prioritize based on count or a default (e.g., semicolon)
        # For now, let's default to semicolon if both are found, or raise an error for ambiguity
        # This case might indicate a malformed header or a need for more specific rules.
        # Based on the example, semicolon is the target for this fix.
        # A simple heuristic: if one is significantly more frequent, use that.
        # For this specific fix, if blue.csv uses ;, we prioritize it.
        # If the problem was about a file using ,, this logic might need to be different.
        # Let's assume for now that if both are present, we prefer the one that is NOT the other.
        # This logic is a bit convoluted. A clearer way:
        if raw_header_content_after_hash.count(';') >= raw_header_content_after_hash.count(','):
             delimiter = ';' # Prioritize semicolon if counts are equal or greater
        else:
             delimiter = ',',
    elif not comma_present and not semicolon_present:
        # This case should be caught by the initial check in the loop, but as a safeguard:
        raise ValueError(f"No known delimiter (',' or ';') found in identified header line: '{raw_header_content_after_hash}'")
    else:
        # Fallback or unhandled case, though logic above should cover known states.
        # This implies one is true and the other is false, which is covered by the first two elifs.
        # To be safe, if we reach here, it means an unexpected combination or logic error.
        raise ValueError(f"Could not reliably determine delimiter for header in {file_path}: '{raw_header_content_after_hash}'")

    header_cols = [col.strip() for col in raw_header_content_after_hash.split(delimiter)]

    # Validate header: ensure no empty column names after stripping,
    # which could happen with malformed lines like "#col1,,col2" or "#col1,col2,"
    if not all(header_cols) or len(header_cols) == 0:
        raise ValueError(f"Malformed or empty header line in {file_path} after processing: '{raw_header_content_after_hash}' resulted in {header_cols}")

    data_rows = []
    # Use csv.reader for robust parsing of data rows, starting from the line after the header
    # csv.reader expects an iterable that yields strings (each line).
    if header_idx + 1 < len(lines):
        # Pass the determined delimiter to csv.reader
        csv_parser = csv.reader(lines[header_idx + 1:], delimiter=delimiter)
        for row_values in csv_parser:
            # Ensure the row is not entirely empty fields, which can happen with blank lines
            # or rows with only delimiters.
            if any(field.strip() for field in row_values):
                data_rows.append(list(row_values))  # Ensure it's a list of strings

    # It's possible to have a header but no data rows, which is valid.
    return [header_cols] + data_rows


if __name__ == "__main__":
    # Example usage:
    # Replace with the actual path to your CSV file if running this script directly.
    # For example, from the root 'raceline_editor' directory:
    # example_file = 'example_trajectories/blue.csv'

    # Relative path from raceline_editor/raceline_editor/raceline/ to example_trajectories/blue.csv
    example_file = "../../example_trajectories/blue.csv"

    raceline_data = import_raceline_csv(example_file)

    if raceline_data:
        print(f"Successfully imported {len(raceline_data)} rows from {example_file}")
        # Print the header and the first 5 rows as a sample
        if len(raceline_data) > 0:
            print("Header:", raceline_data[0])
        for i, row in enumerate(raceline_data[1:6]):
            print(f"Row {i+1}:", row)
    else:
        print(f"Failed to import data from {example_file}")
